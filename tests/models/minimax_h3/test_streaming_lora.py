import ast
import importlib.util
import sys
import types
from abc import ABCMeta, abstractmethod
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors.torch import save_file

ROOT = Path(__file__).parents[3]


def load_module(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(monkeypatch):
    helpers = load_module("h3_stream_lora_test_helpers", Path(__file__).with_name("test_model_disk_streaming.py"), monkeypatch)
    modules = helpers.h3_model_modules.__wrapped__(monkeypatch)
    monkeypatch.setattr(modules[3], "AI_DEVICE", "cpu")
    model = modules[-1].MiniMaxH3Model
    streaming = load_module("lightx2v.models.networks.minimax_h3.streaming_lora", ROOT / "lightx2v/models/networks/minimax_h3/streaming_lora.py", monkeypatch)
    # Execute the actual production MMWeight classes, excluding optional CUDA
    # kernel imports. Only base checkpoint I/O is replaced by the existing tiny
    # fixture; register_lora/apply/apply_lora/remove_lora remain production code.
    scope = {"torch": torch, "ABCMeta": ABCMeta, "abstractmethod": abstractmethod, "logger": logger, "AI_DEVICE": "cpu"}
    for file, names in [
        ("lightx2v/common/ops/utils.py", {"build_lora_and_diff_names"}),
        ("lightx2v/common/ops/mm/mm_weight.py", {"MMWeightTemplate", "MMWeight"}),
    ]:
        tree = ast.parse((ROOT / file).read_text())
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        for node in nodes:
            node.decorator_list = []
        exec(compile(ast.Module(body=nodes, type_ignores=[]), file, "exec"), scope)  # noqa: S102 - execute trusted local production classes

    class Linear(scope["MMWeight"]):
        load = helpers._FakeLinear.load
        load_state_dict = helpers._FakeLinear.load_state_dict

    registry = sys.modules["lightx2v.utils.registry_factory"].MM_WEIGHT_REGISTER
    monkeypatch.setitem(registry, "Default", Linear)
    monkeypatch.setitem(registry, "Default-ForceFp32", Linear)
    return helpers, modules, model, streaming, Linear


def fixture_tensors():
    return {
        "base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.tensor([[1, 2, 3]], dtype=torch.bfloat16),
        "base_model.model.transformer_blocks.0.attn.to_q.lora_B.default.weight": torch.tensor([[1], [2]], dtype=torch.bfloat16),
        "transformer_blocks.1.attn.to_k.lora_down.weight": torch.tensor([[2, 1, 0], [0, 1, 2]], dtype=torch.bfloat16),
        "transformer_blocks.1.attn.to_k.lora_up.weight": torch.tensor([[1, 0], [0, 1]], dtype=torch.bfloat16),
        "transformer_blocks.1.attn.to_k.alpha": torch.tensor(8.0),
        "token_refiner.refiner_blocks.0.attn.to_q.lora_A.weight": torch.ones(1, 3, dtype=torch.bfloat16),
        "token_refiner.refiner_blocks.0.attn.to_q.lora_B.weight": torch.ones(2, 1, dtype=torch.bfloat16),
    }


def make_index(tmp_path, setup, tensors=None, **kwargs):
    _, _, model, streaming, _ = setup
    tensors = fixture_tensors() if tensors is None else tensors
    path = tmp_path / "lora.safetensors"
    save_file(tensors, path)
    shapes = {model._normalize_dynamic_lora_key(key).removesuffix(".lora_down.weight") + ".weight": (2, 3) for key in fixture_tensors() if "lora_A" in key or "lora_down" in key}
    kwargs = {"alpha": 4, "strength": 0.5, **kwargs}
    return streaming.MiniMaxH3StreamingLora(path, normalize_key=model._normalize_dynamic_lora_key, target_shapes=shapes, **kwargs)


def test_index_metadata_only_and_alpha_precedence(tmp_path, setup, monkeypatch):
    streaming = setup[3]
    original = streaming.safe_open
    reads = []

    class Reader:
        def __init__(self, *args, **kwargs):
            self.context = original(*args, **kwargs)

        def __enter__(self):
            self.source = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def keys(self):
            return self.source.keys()

        def get_slice(self, key):
            return self.source.get_slice(key)

        def get_tensor(self, key):
            reads.append(key)
            assert key.endswith(".alpha"), "Index construction read a factor tensor"
            return self.source.get_tensor(key)

    monkeypatch.setattr(streaming, "safe_open", Reader)
    index = make_index(tmp_path, setup)
    assert len(index.pairs) == 3
    assert set(index.blocks) == {0, 1}
    assert len(index.resident) == 1
    assert index.pairs["transformer_blocks.0.attn.to_q.weight"].alpha == 4
    assert index.pairs["transformer_blocks.1.attn.to_k.weight"].alpha == 8
    assert {pair.rank for pair in index.pairs.values()} == {1, 2}
    assert reads == ["transformer_blocks.1.attn.to_k.alpha"]
    assert not index._resident_cpu


@pytest.mark.parametrize("problem", ["missing", "shape", "orphan_alpha", "collision", "unsupported", "alpha", "rank", "factor_dtype"])
def test_invalid_checkpoint_rejected(tmp_path, setup, problem):
    tensors = fixture_tensors()
    a = next(iter(tensors))
    if problem == "missing":
        tensors.pop(a)
    elif problem == "shape":
        tensors[a] = torch.ones(1, 4)
    elif problem == "orphan_alpha":
        tensors["unknown.alpha"] = torch.tensor(1.0)
    elif problem == "collision":
        tensors["transformer_blocks.0.attn.to_q.lora_down.weight"] = tensors[a].clone()
    elif problem == "unsupported":
        tensors["unknown"] = torch.tensor(1.0)
    elif problem == "alpha":
        tensors["transformer_blocks.1.attn.to_k.alpha"] = torch.tensor(float("nan"))
    elif problem == "rank":
        tensors[a] = torch.ones(2, 3)
    elif problem == "factor_dtype":
        tensors[a] = tensors[a].long()
    with pytest.raises(ValueError):
        make_index(tmp_path, setup, tensors)


@pytest.mark.parametrize("alpha", [None, 0, -1, float("inf")])
def test_missing_or_invalid_config_alpha(tmp_path, setup, alpha):
    with pytest.raises(ValueError, match="alpha"):
        make_index(tmp_path, setup, alpha=alpha)


def test_selective_reads_reuse_math_release_and_refiner(tmp_path, setup, monkeypatch):
    helpers, modules, Model, streaming, _ = setup
    _, pre_module, post_module, transformer_module, *_ = modules
    helpers._write_fake_checkpoint(tmp_path, pre_module, post_module, transformer_module)
    index = make_index(tmp_path, setup)
    config = helpers._config(tmp_path, lora_dynamic_apply=True)
    model = Model(str(tmp_path), config, torch.device("cpu"), lora_path=index.path, lora_strength=0.5, lora_alpha=4)
    weights = model.transformer_weights
    index = weights.streaming_lora
    assert len(index.pairs) == 3
    block = weights.streaming_block
    base_pointers = {name: weight.weight.data_ptr() for name, weight in index.weights(block).items()}
    original = streaming.safe_open
    reads = []

    class Reader:
        def __init__(self, *args, **kwargs):
            self.context = original(*args, **kwargs)

        def __enter__(self):
            self.source = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def get_tensor(self, key):
            reads.append(key)
            return self.source.get_tensor(key)

    monkeypatch.setattr(streaming, "safe_open", Reader)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda *_args, **_kwargs: pytest.fail("streamed LoRA must not pin factors"))
    for i, name in [(0, "transformer_blocks.0.attn.to_q.weight"), (1, "transformer_blocks.0.attn.to_k.weight")]:
        reads.clear()
        assert weights.load_streaming_block(i) is block
        linears = index.weights(block)
        assert {key for key, value in linears.items() if value.has_lora_branch} == {name}
        assert len(reads) == 2
        assert all(f"transformer_blocks.{i}." in key for key in reads)
        assert {key: value.weight.data_ptr() for key, value in linears.items()} == base_pointers
        linear = linears[name]
        x = torch.tensor([[1, 2, 3], [-1, 0, 2]], dtype=torch.bfloat16)
        pair = next(iter(index.blocks[i].values()))
        factors = fixture_tensors()
        expected = x @ linear.weight + 0.5 * (pair.alpha / pair.rank) * ((x @ factors[pair.down_key].T) @ factors[pair.up_key].T)
        assert torch.equal(linear.apply(x), expected)
        assert not linear.lora_down.is_pinned()
    refiner = model.pre_weight
    # Tiny fixture's to_cuda is a no-op: explicitly activate its CPU weight.
    for linear in index.weights(refiner).values():
        if getattr(linear, "weight", None) is None:
            linear.weight = linear.pin_weight
    reads.clear()
    for _ in range(2):
        with index.resident_scope(refiner):
            assert sum(weight.has_lora_branch for weight in index.weights(refiner).values()) == 1
        assert not any(weight.has_lora_branch for weight in index.weights(refiner).values())
    assert len(reads) == 2  # Cached on CPU only, read once across evaluations.
    assert all(tensor.device.type == "cpu" for tensors in index._resident_cpu.values() for tensor in tensors)
    weights.release_disk_streaming_buffer()
    assert weights.streaming_block is None
    assert not any(weight.has_lora_branch for weight in index.weights(block).values())
    assert all(not hasattr(weight, "lora_down") and not hasattr(weight, "lora_scale") for weight in index.weights(block).values())
    new_block = weights.load_streaming_block(0)
    assert new_block is not block
    assert sum(weight.has_lora_branch for weight in index.weights(new_block).values()) == 1
    weights.release_disk_streaming_buffer()


def test_resident_cleanup_on_failure(tmp_path, setup):
    index = make_index(tmp_path, setup)
    Linear = setup[-1]
    weight = Linear("token_refiner.refiner_blocks.0.attn.to_q.weight", lora_prefix="token_refiner")
    weight.weight = torch.ones(3, 2, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="pre-infer failed"), index.resident_scope(weight):
        assert weight.has_lora_branch
        raise RuntimeError("pre-infer failed")
    assert not weight.has_lora_branch


def test_merged_streaming_rejected_before_loading(tmp_path, setup):
    helpers, _, Model, _, _ = setup
    with pytest.raises(NotImplementedError, match="dynamic LoRA"):
        Model(str(tmp_path), helpers._config(tmp_path), torch.device("cpu"), lora_path="unused.safetensors")


def test_no_lora_streaming_unchanged(tmp_path, setup):
    helpers, modules, Model, _, _ = setup
    _, pre, post, transformer, *_ = modules
    helpers._write_fake_checkpoint(tmp_path, pre, post, transformer)
    model = Model(str(tmp_path), helpers._config(tmp_path), torch.device("cpu"))
    assert model.transformer_weights.streaming_lora is None
    block = model.transformer_weights.load_streaming_block(1)
    assert block is model.transformer_weights.load_streaming_block(0)
    model.transformer_weights.release_disk_streaming_buffer()


def test_official_raw_adapter_and_lora_compose(tmp_path, setup, monkeypatch):
    helpers, _modules, Model, streaming, _ = setup
    raw_helpers = load_module("h3_stream_lora_raw_helpers", Path(__file__).with_name("test_checkpoint_adapter.py"), monkeypatch)
    _, raw, _ = raw_helpers.write_raw(tmp_path)
    tensors = {}
    for i in (0, 1):
        tensors[f"transformer_blocks.{i}.attn.to_q.lora_A.weight"] = torch.ones(1, 3, dtype=torch.bfloat16) * (i + 1)
        tensors[f"transformer_blocks.{i}.attn.to_q.lora_B.weight"] = torch.ones(4, 1, dtype=torch.bfloat16)
    path = tmp_path / "raw_lora.safetensors"
    save_file(tensors, path)
    model = Model(str(tmp_path), helpers._config(tmp_path, lora_dynamic_apply=True), torch.device("cpu"), lora_path=str(path), lora_alpha=4)
    weights = model.transformer_weights
    assert weights.checkpoint.selected_reader is not None
    pointer = None
    for i in (0, 1, 0):
        block = weights.load_streaming_block(i)
        q = streaming.MiniMaxH3StreamingLora.weights(block)["transformer_blocks.0.attn.to_q.weight"]
        if pointer is None:
            pointer = q.weight.data_ptr()
        assert q.weight.data_ptr() == pointer
        raw_qkv = raw[f"blocks.{i}.attn.qkv_proj.weight"]
        expected_base = torch.vstack((raw_qkv[:2], raw_qkv[6:8])).T
        assert torch.equal(q.weight, expected_base)
        x = torch.tensor([[1, 0, -1], [1, 2, 3]], dtype=torch.bfloat16)
        a, b = tensors[f"transformer_blocks.{i}.attn.to_q.lora_A.weight"], tensors[f"transformer_blocks.{i}.attn.to_q.lora_B.weight"]
        assert torch.equal(q.apply(x), x @ expected_base + 4 * ((x @ a.T) @ b.T))
    # FP32 sensitive heads are not silently given BF16 LoRA factors.
    shapes = streaming.streaming_target_shapes(weights.checkpoint, block, model.pre_weight, model.post_weight)
    assert "proj_in.weight" not in shapes
    assert "proj_out.weight" not in shapes
    weights.release_disk_streaming_buffer()


def test_ordinary_dynamic_loader_and_mmweight_contract(tmp_path, setup, monkeypatch):
    _, _, Model, _, Linear = setup
    index = make_index(tmp_path, setup)
    model = object.__new__(Model)
    model.config = {"dit_disk_streaming": False}
    model.device = torch.device("cpu")
    model.lora_alpha = 4
    model._h3_weight_shapes = {name: (2, 3) for name in index.pairs}
    model.use_tp = False
    # The unchanged ordinary loader pins CPU tensors; avoid requiring a GPU
    # in this regression and verify that it still takes its original path.
    pins = []
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: pins.append(tensor.shape) or tensor)
    loaded = model._load_lora_file(index.path)
    assert pins
    q = Linear("transformer_blocks.0.attn.to_q.weight", lora_prefix="transformer_blocks")
    q.weight = torch.ones(3, 2, dtype=torch.bfloat16)
    q.register_lora(loaded, 0.5)
    x = torch.ones(2, 3, dtype=torch.bfloat16)
    expected = q.apply(x)
    q.remove_lora()
    index.load_block(q, 0)
    assert torch.equal(q.apply(x), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dynamic_math_non_integer_scaling(tmp_path, setup, dtype):
    _, _, _, _, Linear = setup
    gen = torch.Generator().manual_seed(123)
    tensors = {
        "transformer_blocks.0.attn.to_q.lora_down.weight": torch.randn(3, 3, generator=gen, dtype=dtype),
        "transformer_blocks.0.attn.to_q.lora_up.weight": torch.randn(2, 3, generator=gen, dtype=dtype),
    }
    index = make_index(tmp_path, setup, tensors, dtype=dtype, alpha=5, strength=0.7)
    weight = Linear("transformer_blocks.0.attn.to_q.weight", lora_prefix="transformer_blocks")
    weight.weight = torch.randn(3, 2, generator=gen, dtype=dtype)
    x = torch.randn(7, 3, generator=gen, dtype=dtype)
    index.load_block(weight, 0)
    a, b = tensors.values()
    # Preserve the released dynamic path's scalar dtype and operation order.
    scale = torch.tensor(5, dtype=dtype) / 3
    expected = x @ weight.weight + 0.7 * scale * ((x @ a.T) @ b.T)
    assert torch.equal(weight.apply(x), expected)
    index.clear(weight)


def test_production_pre_infer_scope_and_error_cleanup(tmp_path, setup):
    helpers, modules, Model, _streaming, _ = setup
    _, pre, post, transformer, *_ = modules
    helpers._write_fake_checkpoint(tmp_path, pre, post, transformer)
    index = make_index(tmp_path, setup)
    model = Model(str(tmp_path), helpers._config(tmp_path, lora_dynamic_apply=True), torch.device("cpu"), lora_path=index.path, lora_alpha=4)
    for weight in index.weights(model.pre_weight).values():
        weight.weight = weight.pin_weight
    calls = []

    def pre_infer(root, prompt):
        assert any(weight.has_lora_branch for weight in index.weights(root).values())
        calls.append("pre")
        return prompt

    def transformer_infer(weights, pre):
        assert not any(weight.has_lora_branch for weight in index.weights(model.pre_weight).values())
        calls.append("transformer")
        return pre

    def post_infer(root, hidden, pre):
        calls.append("post")
        return hidden

    model.pre_infer = types.SimpleNamespace(infer=pre_infer)
    model.transformer_infer = types.SimpleNamespace(infer=transformer_infer)
    model.post_infer = types.SimpleNamespace(infer=post_infer)
    prompt = torch.ones(1, 3, dtype=torch.bfloat16)
    inputs = {"text_encoder_output": {"prompt_embeds": prompt}}
    assert model._infer_cond_uncond(inputs) is prompt
    assert calls == ["pre", "transformer", "post"]

    def fail(root, prompt):
        pre_infer(root, prompt)
        raise RuntimeError("pre-infer failed")

    model.pre_infer.infer = fail
    with pytest.raises(RuntimeError, match="pre-infer failed"):
        model._infer_cond_uncond(inputs)
    assert not any(weight.has_lora_branch for weight in index.weights(model.pre_weight).values())
    with pytest.raises(NotImplementedError, match="selective LoRA index"):
        model._load_lora_file(index.path)
    with pytest.raises(NotImplementedError, match="runtime adapter switching"):
        model._update_lora(index.path, 1)
    model.transformer_weights.release_disk_streaming_buffer()


def test_ordinary_merged_lora_contract(tmp_path, setup, monkeypatch):
    index = make_index(tmp_path, setup)
    adapter_base = types.ModuleType("lightx2v.models.networks.lora_adapter")
    adapter_base.LoraAdapter = object
    monkeypatch.setitem(sys.modules, adapter_base.__name__, adapter_base)
    module = load_module("h3_ordinary_lora_regression", ROOT / "lightx2v/models/networks/minimax_h3/lora.py", monkeypatch)
    adapter = module.MiniMaxH3LoraAdapter()
    base = {name: torch.ones(2, 3, dtype=torch.bfloat16) for name in index.pairs}
    adapter.model = types.SimpleNamespace(config={"lora_merge_device": "cpu"}, use_tp=False, original_weight_dict=base)
    assert adapter._merge_file(index.path, strength=0.5, alpha=4) == 3
    fixture = fixture_tensors()
    for name, pair in index.pairs.items():
        expected = torch.ones(2, 3, dtype=torch.bfloat16)
        expected.add_(fixture[pair.up_key] @ fixture[pair.down_key], alpha=0.5 * pair.alpha / pair.rank)
        assert torch.equal(base[name], expected)
