import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from test_model_disk_streaming import _config, h3_model_modules  # noqa: F401
from test_transformer_disk_streaming import h3_modules  # noqa: F401

ROOT = Path(__file__).parents[3]
spec = importlib.util.spec_from_file_location("h3_checkpoint_adapter_under_test", ROOT / "lightx2v/models/networks/minimax_h3/checkpoint.py")
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


def write_raw(path):
    config = {
        "hidden_size": 3,
        "num_layers": 2,
        "token_refiner_num_layers": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 2,
        "ffn_hidden_size": 4,
        "latents_dim": 1,
        "audio_latents_dim": 2,
        "patch_size": [1, 1, 2],
        "text_dim": 5,
        "timestep_input_dim": 4,
        "time_embed_hidden_size": 3,
        "time_embed_dim": 2,
        "rope_inv_freq_len": 2,
    }
    schema = C._official_schema(C._native_config(config))
    tensors = {}
    for i, (name, (_, shape, dtype, _)) in enumerate(schema.items()):
        tensor = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape) + i
        tensors[name] = tensor.to(torch.float32 if dtype == "F32" else torch.bfloat16)
    tensors["rope.inv_freq"] = 1.0 / (10000.0 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4))
    weight_map = {name: f"shard-{i % 2}.safetensors" for i, name in enumerate(sorted(tensors))}
    for shard in set(weight_map.values()):
        save_file({k: v for k, v in tensors.items() if weight_map[k] == shard}, path / shard)
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return config, tensors, weight_map


@pytest.fixture
def raw(tmp_path):
    config, tensors, weight_map = write_raw(tmp_path)
    plan = C.MiniMaxH3CheckpointPlan(tmp_path)
    return config, tensors, weight_map, plan, C.MiniMaxH3SelectedSourceReader(plan, row_chunk_size=2)


@pytest.fixture(params=["cpu", "mps"])
def destination_device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    return request.param


def destinations(plan, names, transpose=False, device="cpu"):
    return {
        name: (
            torch.empty(
                tuple(reversed(plan.targets[name][1].shape)) if transpose else plan.targets[name][1].shape,
                dtype=torch.float32 if plan.targets[name][1].dtype == "F32" else torch.bfloat16,
                device=device,
            ),
            transpose,
        )
        for name in names
    }


def test_official_detection_and_complete_config_driven_plan(raw):
    _, tensors, _, plan, _ = raw
    assert plan.format == "official_raw"
    assert plan.block_indices == (0, 1)
    assert set(plan.entries) == set(tensors)
    C.validate_mapping(plan.entries.values(), tensors, plan.targets)
    assert len(plan.entries) == 47  # Tiny config, not the release's 535/638 counts.
    assert len(plan.targets) == 52
    assert plan.entries["rope.inv_freq"].targets == ()


@pytest.mark.parametrize("name", ["transformer_blocks.0.attn.to_q.weight", "proj_in.weight"])
def test_native_format_not_misclassified(tmp_path, name):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "absent.safetensors"}}))
    assert C.MiniMaxH3CheckpointPlan(tmp_path).format == "native"


def test_mixed_format_rejected(raw, tmp_path):
    mapping = dict(raw[2], **{"transformer_blocks.0.attn.to_q.weight": "shard-0.safetensors"})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError, match="mixed"):
        C.MiniMaxH3CheckpointPlan(tmp_path)


@pytest.mark.parametrize("prefix,target", [("blocks.0", "transformer_blocks.0"), ("token_refiner.blocks.0", "token_refiner.refiner_blocks.0")])
def test_qkv_head_interleave_and_transpose(raw, prefix, target, destination_device):
    _, tensors, _, plan, reader = raw
    names = [f"{target}.attn.to_{q}.weight" for q in "qkv"]
    raw_qkv = tensors[f"{prefix}.attn.qkv_proj.weight"]
    for transpose in (False, True):
        dest = destinations(plan, names, transpose, device=destination_device)
        reader.write_targets(dest)
        for component, name in enumerate(names):
            expected = torch.vstack([raw_qkv[component * 2 : component * 2 + 2], raw_qkv[6 + component * 2 : 8 + component * 2]])
            actual = (dest[name][0].t() if transpose else dest[name][0]).cpu()
            assert actual.shape == (4, 3)
            assert torch.equal(actual, expected)
            assert not torch.equal(actual, raw_qkv.chunk(3, dim=0)[component])


def test_slice_only_reading_and_owned_slice_storage(raw, monkeypatch):
    _, _, _, plan, reader = raw
    reads = []

    class Slice:
        def __init__(self, inner, name):
            self.inner, self.name = inner, name

        def get_shape(self):
            return self.inner.get_shape()

        def get_dtype(self):
            return self.inner.get_dtype()

        def __getitem__(self, index):
            reads.append((self.name, index.start, index.stop))
            return self.inner[index]

    class Open:
        def __init__(self, *args, **kwargs):
            self.inner = safe_open(*args, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def get_slice(self, name):
            return Slice(self.inner.get_slice(name), name)

        def get_tensor(self, name):
            pytest.fail("adapter must not get_tensor")

    monkeypatch.setattr(C, "safe_open", Open)
    name = "blocks.0.attn.qkv_proj.weight"
    sl = reader.read_source_slice(name, 2, 4)
    assert sl.untyped_storage().nbytes() == 2 * 3 * 2
    reads.clear()
    dest = destinations(plan, [t.name for t in plan.entries[name].targets])
    reader.write_targets(dest)
    assert reads == [(name, i, i + 2) for i in range(0, 12, 2)]


@pytest.mark.parametrize("prefix,target", [("blocks.0", "transformer_blocks.0"), ("token_refiner.blocks.0", "token_refiner.refiner_blocks.0")])
def test_fc1_swap_without_full_cat(raw, monkeypatch, prefix, target, destination_device):
    _, tensors, _, plan, reader = raw
    name = target + ".ff.net.0.proj.weight"
    source = tensors[prefix + ".mlp.fc1.weight"]
    dest = destinations(plan, [name], transpose=True, device=destination_device)
    monkeypatch.setattr(torch, "cat", lambda *a, **k: pytest.fail("no full fused cat"))
    reader.write_targets(dest)
    actual = dest[name][0].t().cpu()
    assert torch.equal(actual[:4], source[4:])
    assert torch.equal(actual[4:], source[:4])


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_written_targets_outlive_source_and_own_storage(raw, destination_device, transpose, strided):
    _, tensors, _, plan, reader = raw
    mapping = {
        "proj_in.weight": "video_patch_proj.weight",
        "context_embedder.weight": "condition_proj.weight",
        "transformer_blocks.0.attn.to_out.0.weight": "blocks.0.attn.out_proj.weight",
        "norm_out.norm.weight": "final_layer.norm.weight",
    }
    dest = {}
    expected = {}
    for target, source in mapping.items():
        value = tensors[source]
        transposed = transpose and value.ndim == 2
        expected[target] = value.t() if transposed else value
        if strided:
            tensor = torch.empty((*expected[target].shape, 2), dtype=value.dtype, device=destination_device)[..., 0]
        else:
            tensor = torch.empty_like(value, device=destination_device)
            if transposed:
                tensor = tensor.t()
        dest[target] = (tensor, transposed)
    pointers = {name: tensor.data_ptr() for name, (tensor, _) in dest.items()}
    reader.write_targets(dest)
    # write_targets has closed every source context; only destinations survive.
    del reader
    for target, (tensor, _) in dest.items():
        assert tensor.data_ptr() == pointers[target]
        assert torch.equal(tensor.cpu(), expected[target])
        tensor.zero_()
    # Mutating a destination must not modify or alias the checkpoint storage.
    reader = C.MiniMaxH3SelectedSourceReader(plan, row_chunk_size=2)
    reader.write_targets(dest)
    for target, (tensor, _) in dest.items():
        assert tensor.data_ptr() == pointers[target]
        assert torch.equal(tensor.cpu(), expected[target])


def test_norm_refiner_and_nonblock_mapping_and_fp32(raw):
    _, tensors, _, plan, reader = raw
    names = plan.non_block_tensor_names()
    reader.validate_sources(names)
    targets = [t.name for name in names for t in plan.entries[name].targets]
    dest = destinations(plan, targets)
    reader.write_targets(dest)
    for source, target in [
        ("video_patch_proj", "proj_in"),
        ("audio_patch_proj", "audio_proj_in"),
        ("condition_proj", "context_embedder"),
        ("time_embedder.proj_in", "time_embedder.linear_1"),
        ("time_embedder.proj_out", "time_embedder.linear_2"),
        ("final_layer.adaln_proj.linear", "norm_out.linear"),
        ("final_layer.video_out", "proj_out"),
        ("final_layer.audio_out", "audio_proj_out"),
    ]:
        for suffix in ("weight", "bias"):
            assert torch.equal(dest[f"{target}.{suffix}"][0], tensors[f"{source}.{suffix}"])
            assert dest[f"{target}.{suffix}"][0].dtype == tensors[f"{source}.{suffix}"].dtype
    for q in "qk":
        assert torch.equal(dest[f"token_refiner.refiner_blocks.0.attn.norm_{q}.weight"][0], tensors[f"token_refiner.blocks.0.attn.{q}_norm.weight"])
    assert torch.equal(dest["norm_out.norm.weight"][0], tensors["final_layer.norm.weight"])
    assert torch.equal(dest["token_refiner.final_norm.weight"][0], tensors["token_refiner.final_norm.weight"])


@pytest.mark.parametrize("failure", ["shape", "dtype", "rope_value"])
def test_invalid_real_header_or_rope_fails(raw, tmp_path, failure):
    _, tensors, weight_map, _, reader = raw
    name = "rope.inv_freq" if failure == "rope_value" else "blocks.0.attn.qkv_proj.weight"
    tensors[name] = tensors[name][:-1] if failure == "shape" else tensors[name].float() if failure == "dtype" else tensors[name] + 1
    shard = weight_map[name]
    save_file({k: v for k, v in tensors.items() if weight_map[k] == shard}, tmp_path / shard)
    with pytest.raises(ValueError, match="rope.inv_freq" if failure == "rope_value" else failure + " mismatch"):
        reader.validate_sources([name])


def test_fp32_special_parameter_must_not_be_bf16(raw, tmp_path):
    _, tensors, weight_map, _, reader = raw
    name = "video_patch_proj.weight"
    tensors[name] = tensors[name].bfloat16()
    shard = weight_map[name]
    save_file({k: v for k, v in tensors.items() if weight_map[k] == shard}, tmp_path / shard)
    with pytest.raises(ValueError, match="dtype mismatch"):
        reader.validate_sources([name])


def test_wrong_destination_orientation_or_dtype_fails_before_copy(raw):
    reader = raw[4]
    name = "transformer_blocks.0.attn.to_q.weight"
    with pytest.raises(ValueError, match="destination shape/dtype"):
        reader.write_targets({name: (torch.zeros(4, 3, dtype=torch.bfloat16), True)})
    with pytest.raises(ValueError, match="destination shape/dtype"):
        reader.write_targets({name: (torch.zeros(4, 3, dtype=torch.float32), False)})
    with pytest.raises(KeyError, match="missing target plan"):
        reader.write_targets({"unknown": (torch.zeros(1), False)})


@pytest.mark.parametrize("failure", ["unknown", "missing"])
def test_source_coverage_rejected(raw, tmp_path, failure):
    mapping = dict(raw[2])
    if failure == "unknown":
        mapping["rope.not_allowed"] = "shard-0.safetensors"
    else:
        del mapping["blocks.0.norm1.weight"]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError, match=failure):
        C.MiniMaxH3CheckpointPlan(tmp_path)


def test_duplicate_and_missing_target_coverage_rejected(raw):
    plan = raw[3]
    entries = list(plan.entries.values())
    first = next(i for i, e in enumerate(entries) if e.targets)
    entry = entries[first]
    entries[first] = entry._replace(targets=entry.targets + entry.targets)
    with pytest.raises(ValueError, match="duplicate target"):
        C.validate_mapping(entries, plan.entries, plan.targets)
    entries[first] = entry._replace(targets=())
    with pytest.raises(ValueError, match="target mapping mismatch: missing"):
        C.validate_mapping(entries, plan.entries, plan.targets)


def test_cross_shard_block_load_and_missing_shard_fail(raw, tmp_path):
    _, _, _, plan, reader = raw
    sources = plan.tensor_names_for_block(0)
    assert len(sources) == 10
    assert len(plan.shards_for_sources(sources)) == 2
    dest = destinations(plan, [t.name for n in sources for t in plan.entries[n].targets])
    reader.write_targets(dest)
    (tmp_path / "shard-1.safetensors").unlink()
    assert C.MiniMaxH3CheckpointPlan(tmp_path).block_indices == (0, 1)
    with pytest.raises(FileNotFoundError, match="requested shard missing"):
        reader.write_targets(dest)
    with pytest.raises(FileNotFoundError, match="missing shard files"):
        C.MiniMaxH3ShardCheckpoint(tmp_path)


def test_official_streaming_reuses_storage_and_recreates(tmp_path, monkeypatch, h3_modules):
    _, weights_module, _ = h3_modules
    config, tensors, _ = write_raw(tmp_path)
    config.update(C._native_config(config), dit_disk_streaming=True, dit_original_ckpt=str(tmp_path))
    monkeypatch.setattr(weights_module, "AI_DEVICE", "cpu")
    weights = weights_module.MiniMaxH3TransformerWeights(config)
    block = weights.streaming_block
    records = {}
    for name, attr, _ in weights_module._iter_base_attrs(block):
        records[name] = attr

    def buffers():
        stack, result = [weights.streaming_block], {}
        while stack:
            module = stack.pop()
            for name, attr, _ in getattr(module, "base_attrs", ()):
                tensor = getattr(module, attr)
                assert tensor is getattr(module, attr + "_cuda_buffer")
                result[name] = (id(tensor), tensor.data_ptr())
            stack.extend(getattr(module, "_modules", {}).values())
        return result

    before = buffers()
    assert weights.load_streaming_block(1) is block
    assert buffers() == before
    assert torch.equal(block.ff.in_proj.weight[:, :4].t(), tensors["blocks.1.mlp.fc1.weight"][4:])
    weights.release_disk_streaming_buffer()
    assert weights.streaming_block is None and block.attn.to_q.weight is None
    new = weights.load_streaming_block(0)
    assert new is not block
    before = buffers()
    assert weights.load_streaming_block(1) is new and buffers() == before


def test_raw_config_drives_native_dimensions_without_runtime_defaults(tmp_path, monkeypatch, h3_modules):
    _, weights_module, _ = h3_modules
    write_raw(tmp_path)
    monkeypatch.setattr(weights_module, "AI_DEVICE", "cpu")
    weights = weights_module.MiniMaxH3TransformerWeights({"dit_disk_streaming": True, "dit_original_ckpt": str(tmp_path)})
    assert weights.num_layers == 2
    assert weights.config["hidden_size"] == 3
    assert weights.config["num_refiner_layers"] == 1


def test_official_model_prepost_initialization(tmp_path, monkeypatch, h3_model_modules):
    _, _, _, transformer, _, _, model_module = h3_model_modules
    config, tensors, _ = write_raw(tmp_path)
    monkeypatch.setattr(transformer, "AI_DEVICE", "cpu")
    config = _config(tmp_path, **config)
    del config["num_refiner_layers"]
    model = model_module.MiniMaxH3Model(str(tmp_path), config, torch.device("cpu"))
    assert model.config["num_refiner_layers"] == 1
    assert model.config["freq_dim"] == 4
    assert model.pre_weight.proj_in.pin_weight.dtype == torch.float32
    assert model.post_weight.proj_out.pin_weight.dtype == torch.float32
    assert torch.equal(model.pre_weight.context_embedder.pin_weight.t(), tensors["condition_proj.weight"])
    assert torch.equal(model.post_weight.proj_out.pin_weight.t(), tensors["final_layer.video_out.weight"])


def test_quantized_streaming_remains_rejected(tmp_path, h3_modules):
    _, weights_module, _ = h3_modules
    with pytest.raises(NotImplementedError, match="quantized"):
        weights_module.MiniMaxH3TransformerWeights({"dit_disk_streaming": True, "dit_quantized": True})


def test_quantized_nonstreaming_keeps_base_loader(tmp_path, h3_model_modules):
    model_module = h3_model_modules[-1]
    with pytest.raises(AssertionError, match="full checkpoint loading"):
        model_module.MiniMaxH3Model(
            str(tmp_path), _config(tmp_path, dit_disk_streaming=False, dit_quantized=True, dit_quant_scheme="int8-torchao", dit_quantized_ckpt=str(tmp_path)), torch.device("cpu")
        )
