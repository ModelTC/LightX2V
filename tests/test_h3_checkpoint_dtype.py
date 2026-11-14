"""Exercise H3's checkpoint loader on CPU without importing CUDA backends."""

import ast
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")
ROOT = Path(__file__).resolve().parents[1]


def _loader_type():
    parallel_path = ROOT / "lightx2v/models/networks/minimax_h3/parallel.py"
    parallel_tree = ast.parse(parallel_path.read_text())
    functions = [
        node
        for node in parallel_tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"tensor_parallel_split_type", "split_tensor_parallel_weight"}
    ]
    namespace = {"torch": torch, "os": os, "load_file": safetensors.load_file, "GET_DTYPE": lambda: torch.bfloat16}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(parallel_path), "exec"), namespace)

    policy_path = ROOT / "lightx2v/utils/fp8_f16_accum_checkpoint.py"
    policy_tree = ast.parse(policy_path.read_text())
    suffixes = next(node for node in policy_tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "H3_PROJECTION_SUFFIXES" for target in node.targets))
    namespace["H3_PROJECTION_SUFFIXES"] = ast.literal_eval(suffixes.value)

    model_path = ROOT / "lightx2v/models/networks/minimax_h3/model.py"
    model_tree = ast.parse(model_path.read_text())
    model = next(node for node in model_tree.body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Model")
    names = {"_checkpoint_runtime_dtype", "_checkpoint_load_device", "_validate_checkpoint_devices", "_select_tensor_parallel_shard", "_load_safetensor_to_dict"}
    model.body = [node for node in model.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in model.body} == names
    model.bases = []
    model.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[model], type_ignores=[])), str(model_path), "exec"), namespace)
    return namespace["MiniMaxH3Model"]


@pytest.fixture
def make_loader():
    loader_type = _loader_type()

    def make(tp_size=1, tp_rank=0, text_refiner_tensor_parallel=True):
        loader = loader_type()
        loader.device = torch.device("cpu")
        loader.tp_size = tp_size
        loader.tp_rank = tp_rank
        loader.config = {"text_refiner_tensor_parallel": text_refiner_tensor_parallel}
        loader.remove_keys = []
        loader._record_checkpoint_metadata = lambda tensors: setattr(loader, "checkpoint_dtypes", {name: tensor.dtype for name, tensor in tensors.items()})
        return loader

    return make


def _matrix(rows=8, columns=8, dtype=torch.float32):
    return (torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns) / 101).to(dtype)


def _mixed_checkpoint():
    tensors = {}
    for prefix in ("token_refiner.refiner_blocks.0", "transformer_blocks.0"):
        for projection in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.2"):
            tensors[f"{prefix}.{projection}.weight"] = _matrix()
        tensors[f"{prefix}.ff.net.0.proj.weight"] = _matrix(16)
        tensors[f"{prefix}.ff.net.0.proj.bias"] = torch.arange(16, dtype=torch.float32) / 7
        tensors[f"{prefix}.ff.net.2.bias"] = torch.arange(8, dtype=torch.float32) / 7
        tensors[f"{prefix}.norm1.weight"] = torch.linspace(0.75, 1.25, 8)
        tensors[f"{prefix}.attn.norm_q.weight"] = torch.linspace(0.75, 1.25, 8)
    tensors["context_embedder.weight"] = _matrix(dtype=torch.bfloat16)
    tensors["context_embedder.bias"] = torch.zeros(8, dtype=torch.bfloat16)
    for module in ("transformer_blocks.0.adaln_proj.linear", "norm_out.linear"):
        tensors[f"{module}.weight"] = _matrix()
        tensors[f"{module}.bias"] = torch.zeros(8)
    for module in ("proj_in", "audio_proj_in", "time_embedder.linear_1", "time_embedder.linear_2", "proj_out", "audio_proj_out"):
        tensors[f"{module}.weight"] = _matrix()
        tensors[f"{module}.bias"] = torch.zeros(8)
    tensors["norm_out.norm.weight"] = torch.ones(8)
    tensors["token_refiner.final_norm.weight"] = torch.ones(8)
    return tensors


def _save_checkpoint(path, tensors):
    if path.suffix == ".pt":
        torch.save(tensors, path)
    else:
        safetensors.save_file(tensors, path)


def _load_checkpoint(loader, path):
    return loader._load_safetensor_to_dict(str(path), unified_dtype=True, sensitive_layer=set())


@pytest.mark.parametrize("extension", [".pt", ".safetensors"])
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
def test_mixed_checkpoint_runs_bf16_linears_and_preserves_fp32_layers(tmp_path, make_loader, extension, tp_size, tp_rank):
    source = _mixed_checkpoint()
    checkpoint = tmp_path / f"merged{extension}"
    _save_checkpoint(checkpoint, source)
    loader = make_loader(tp_size, tp_rank)
    loaded = _load_checkpoint(loader, checkpoint)

    preserved = {name for name in source if name.startswith(("proj_in.", "audio_proj_in.", "time_embedder.", "proj_out.", "audio_proj_out.")) or ".norm1." in name or ".norm_q." in name or name in {"norm_out.norm.weight", "token_refiner.final_norm.weight"}}
    for name, tensor in loaded.items():
        if name in preserved:
            assert tensor.dtype == torch.float32, name
            torch.testing.assert_close(tensor, source[name], rtol=0, atol=0)
        else:
            assert tensor.dtype == torch.bfloat16, name
            if name.endswith(".weight"):
                bias = loaded.get(name.removesuffix(".weight") + ".bias")
                inputs = torch.ones(3, tensor.shape[1], dtype=torch.bfloat16)
                output = torch.nn.functional.linear(inputs, tensor, bias)
                assert output.dtype == torch.bfloat16
                assert torch.isfinite(output).all()

    # Reading normalizes only runtime tensors; files and source metadata retain
    # the merged checkpoint's original precision.
    disk = torch.load(checkpoint, weights_only=True) if extension == ".pt" else safetensors.load_file(checkpoint)
    assert loader.checkpoint_dtypes == {name: tensor.dtype for name, tensor in source.items()}
    for name in source:
        torch.testing.assert_close(disk[name], source[name], rtol=0, atol=0)


def test_tp_dtype_conversion_preserves_column_row_and_fused_swiglu_shards(tmp_path, make_loader):
    source = _mixed_checkpoint()
    checkpoint = tmp_path / "merged.safetensors"
    _save_checkpoint(checkpoint, source)
    shards = [_load_checkpoint(make_loader(2, rank), checkpoint) for rank in range(2)]

    for prefix in ("token_refiner.refiner_blocks.0", "transformer_blocks.0"):
        column = f"{prefix}.attn.to_q.weight"
        row = f"{prefix}.attn.to_out.0.weight"
        fused = f"{prefix}.ff.net.0.proj.weight"
        fused_bias = f"{prefix}.ff.net.0.proj.bias"
        torch.testing.assert_close(torch.cat([shard[column] for shard in shards], dim=0), source[column].bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(torch.cat([shard[row] for shard in shards], dim=1), source[row].bfloat16(), rtol=0, atol=0)
        for rank, indices in enumerate(([0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15])):
            torch.testing.assert_close(shards[rank][fused], source[fused][indices].bfloat16(), rtol=0, atol=0)
            torch.testing.assert_close(shards[rank][fused_bias], source[fused_bias][indices].bfloat16(), rtol=0, atol=0)


def test_replicated_refiner_still_normalizes_merged_weights(tmp_path, make_loader):
    source = _mixed_checkpoint()
    checkpoint = tmp_path / "merged.safetensors"
    _save_checkpoint(checkpoint, source)
    loaded = _load_checkpoint(make_loader(2, 1, text_refiner_tensor_parallel=False), checkpoint)
    for name, tensor in source.items():
        if name.startswith("token_refiner.") and tensor.ndim == 2:
            torch.testing.assert_close(loaded[name], tensor.bfloat16(), rtol=0, atol=0)


def test_loader_preserves_quantized_scales_lora_and_unrecognized_tensors(tmp_path, make_loader):
    source = {
        "transformer_blocks.0.attn.to_q.weight": _matrix().to(torch.float8_e4m3fn),
        "transformer_blocks.0.attn.to_q.weight_scale": torch.tensor([0.03125]),
        "transformer_blocks.0.attn.to_k.weight": torch.ones(8, 8, dtype=torch.int8),
        "transformer_blocks.0.attn.to_v.lora_down.weight": _matrix(2),
        "transformer_blocks.0.attn.to_v.lora_up.weight": _matrix(8, 2),
        "unknown.weight": _matrix(),
        "unknown.bias": torch.ones(8),
        "token_refiner.refiner_blocks.0.norm1.weight": torch.ones(8, dtype=torch.float16),
        "context_embedder.weight": _matrix(dtype=torch.float16),
        "context_embedder.bias": torch.ones(8, dtype=torch.float16),
    }
    checkpoint = tmp_path / "special.safetensors"
    _save_checkpoint(checkpoint, source)
    loaded = _load_checkpoint(make_loader(), checkpoint)
    for name, tensor in source.items():
        if name.startswith("context_embedder."):
            torch.testing.assert_close(loaded[name], tensor.bfloat16(), rtol=0, atol=0)
        else:
            assert loaded[name].dtype == tensor.dtype, name
            torch.testing.assert_close(loaded[name].float(), tensor.float(), rtol=0, atol=0)


def test_loader_retains_checkpoint_key_filtering(tmp_path, make_loader):
    source = _mixed_checkpoint()
    checkpoint = tmp_path / "merged.safetensors"
    _save_checkpoint(checkpoint, source)
    loader = make_loader()
    loader.remove_keys = [".attn.norm_q."]
    loader.preserved_keys = ["token_refiner."]
    loaded = _load_checkpoint(loader, checkpoint)
    assert set(loaded) == {name for name in source if name.startswith("token_refiner.") and ".attn.norm_q." not in name}
    assert loaded["token_refiner.refiner_blocks.0.attn.to_q.weight"].dtype == torch.bfloat16
