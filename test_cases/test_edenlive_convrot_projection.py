"""ConvRot projection sharing: exact arithmetic, strided output and safe fallback."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

import lightx2v.models.networks.wan.infer.edenlive.convrot_projection as projection_module
from lightx2v.common.ops.mm.mm_weight import MMWeightWint8ConvRot
from lightx2v.models.networks.wan.edenlive_model import EdenLiveModel
from lightx2v.models.networks.wan.infer.edenlive.convrot_projection import ConvRotProjectionGroup, convrot_projection_mm, convrot_projection_quant
from lightx2v.models.networks.wan.infer.edenlive.packed_projection import FP8ProjectionGroup
from lightx2v.models.networks.wan.model import WanModel

cuda_only = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="ConvRot grouped projections require SM120")


@pytest.fixture
def cuda_state():
    pytest.importorskip("comfy_kitchen.backends.cuda")
    torch.manual_seed(123)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _projections(bias=True, dtype=torch.bfloat16, input_dim=256, output_dim=128):
    modules = []
    for name in ("q", "k", "v"):
        module = MMWeightWint8ConvRot(name + ".weight", name + ".bias" if bias else None)
        module.infer_dtype = dtype
        module.weight = torch.randint(-127, 128, (output_dim, input_dim), device="cuda", dtype=torch.int8)
        module.weight_scale = torch.rand(output_dim, 1, device="cuda", dtype=torch.float32).mul_(0.02).add_(0.01)
        module.bias = torch.randn(output_dim, device="cuda", dtype=dtype) if bias else None
        module._convrot_groupsize = 256
        modules.append(module)
    return tuple(modules)


def _assert_equal(expected, actual):
    assert len(actual) == len(expected)
    for reference, output in zip(expected, actual):
        assert torch.equal(reference, output)


@cuda_only
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_eager_fullgraph_exact_and_weights_unchanged(cuda_state, packed, bias, dtype):
    modules = _projections(bias=bias, dtype=dtype)
    weights = tuple(module.weight.clone() for module in modules)
    group = ConvRotProjectionGroup.from_modules(modules, packed=packed)
    assert group is not None
    compiled = torch.compile(group.apply, fullgraph=True)
    with torch.inference_mode():
        for tokens in (3, 17, 67):
            # Column slicing verifies the original contiguous-input conversion.
            x = torch.randn(tokens, 512, device="cuda", dtype=dtype)[:, ::2]
            original_x = x.clone()
            expected = tuple(module.apply(x) for module in modules)
            for apply in (group.apply, compiled):
                actual = apply(x)
                _assert_equal(expected, actual)
                assert all(output.stride() == ((384 if packed else 128), 1) for output in actual)
                assert torch.equal(x, original_x)
    for module, original in zip(modules, weights):
        assert torch.equal(module.weight, original)


@cuda_only
@pytest.mark.parametrize("tokens", [880, 2640])
@pytest.mark.parametrize("packed", [False, True])
def test_production_projection_shapes_exact(cuda_state, tokens, packed):
    modules = _projections(input_dim=3072, output_dim=3072)
    group = ConvRotProjectionGroup.from_modules(modules, packed=packed)
    x = torch.randn(tokens, 3072, device="cuda", dtype=torch.bfloat16)
    _assert_equal(tuple(module.apply(x) for module in modules), group.apply(x))


@cuda_only
@pytest.mark.parametrize("packed", [False, True])
def test_quantization_and_gemm_invocation_counts(cuda_state, monkeypatch, packed):
    modules = _projections()
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    expected = tuple(module.apply(x) for module in modules)
    original_quant, original_mm = projection_module.convrot_projection_quant, projection_module.convrot_projection_mm
    counts = {"quant": 0, "mm": 0}

    def quant(value):
        counts["quant"] += 1
        return original_quant(value)

    def mm(*args):
        counts["mm"] += 1
        return original_mm(*args)

    monkeypatch.setattr(projection_module, "convrot_projection_quant", quant)
    monkeypatch.setattr(projection_module, "convrot_projection_mm", mm)
    group = ConvRotProjectionGroup.from_modules(modules, packed=packed)
    _assert_equal(expected, group.apply(x))
    assert counts == {"quant": 1, "mm": 1 if packed else 3}


@cuda_only
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("change", ["lora", "diff", "weight", "scale", "bias"])
def test_dynamic_changes_after_compile_use_original_modules(cuda_state, packed, change):
    modules = _projections()
    group = ConvRotProjectionGroup.from_modules(modules, packed=packed)
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(group.apply, fullgraph=True)
    compiled(x)
    module = modules[0]
    if change == "lora":
        module.has_lora_branch = True
        module.lora_down = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
        module.lora_up = torch.randn(128, 2, device="cuda", dtype=torch.bfloat16)
        module.lora_strength, module.lora_scale = 1.0, 0.25
    elif change == "diff":
        module.has_diff = True
        module.bias_diff = torch.ones_like(module.bias)
    elif change == "weight":
        module.weight = torch.randint(-127, 128, module.weight.shape, device="cuda", dtype=torch.int8)
    elif change == "scale":
        module.weight_scale = module.weight_scale * 2
    else:
        module.bias = module.bias + 1
    expected = tuple(projection.apply(x) for projection in modules)
    _assert_equal(expected, group.apply(x))
    _assert_equal(expected, compiled(x))


@cuda_only
@pytest.mark.parametrize("tokens", [1, 2])
@pytest.mark.parametrize("packed", [False, True])
def test_small_m_preserves_original_specialized_kernel(cuda_state, tokens, packed, monkeypatch):
    modules = _projections()
    group = ConvRotProjectionGroup.from_modules(modules, packed=packed)
    x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)

    def forbidden(*args):
        pytest.fail("small M must use original specialized ConvRot path")

    monkeypatch.setattr(projection_module, "convrot_projection_quant", forbidden)
    _assert_equal(tuple(module.apply(x) for module in modules), group.apply(x))


def test_cpu_and_wrong_types_are_rejected():
    assert ConvRotProjectionGroup.from_modules(()) is None
    assert ConvRotProjectionGroup.from_modules((object(),)) is None
    module = MMWeightWint8ConvRot("weight", None)
    module.weight = torch.empty(128, 256, dtype=torch.int8)
    assert ConvRotProjectionGroup.from_modules((module,)) is None


@cuda_only
@pytest.mark.parametrize("change", ["lora", "diff", "lazy_load", "create_cuda_buffer", "create_cpu_buffer", "mixed_bias", "wrong_scale", "wrong_dtype", "noncontiguous", "groupsize"])
def test_incompatible_modules_rejected(cuda_state, change):
    modules = _projections()
    module = modules[-1]
    if change == "lora":
        module.has_lora_branch = True
    elif change == "diff":
        module.has_diff = True
    elif change in ("lazy_load", "create_cuda_buffer", "create_cpu_buffer"):
        setattr(module, change, True)
    elif change == "mixed_bias":
        module.bias = None
    elif change == "wrong_scale":
        module.weight_scale = module.weight_scale.to(torch.bfloat16)
    elif change == "wrong_dtype":
        module.weight = module.weight.to(torch.bfloat16)
    elif change == "noncontiguous":
        module.weight = module.weight.t().contiguous().t()
    else:
        module._convrot_groupsize = 64
    assert ConvRotProjectionGroup.from_modules(modules) is None


@cuda_only
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("shape", [(17, 256, 128), (880, 3072, 3072), (2640, 3072, 3072)])
def test_cuda_graph_replay_exact(cuda_state, packed, shape):
    tokens, input_dim, output_dim = shape
    modules = _projections(input_dim=input_dim, output_dim=output_dim)
    group = ConvRotProjectionGroup.from_modules(modules, packed=packed)
    x = torch.randn(tokens, input_dim, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            group.apply(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = group.apply(x)
    x.normal_()
    graph.replay()
    _assert_equal(tuple(module.apply(x) for module in modules), actual)


@cuda_only
@pytest.mark.parametrize("packed", [False, True])
def test_cublas_fallback_preserves_original_epilogue(cuda_state, packed, monkeypatch):
    monkeypatch.setattr(projection_module.kitchen_cuda, "_DISABLE_CUTLASS_INT8", True)
    modules = _projections()
    group = ConvRotProjectionGroup.from_modules(modules, packed=packed)
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    _assert_equal(tuple(module.apply(x) for module in modules), group.apply(x))


@cuda_only
def test_non_cuda_backend_selection_is_respected(cuda_state):
    modules = _projections()
    with projection_module.comfy_kitchen.registry.use_backend("triton"):
        assert ConvRotProjectionGroup.from_modules(modules) is None


@cuda_only
def test_fake_tensor_shape_contract(cuda_state):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(17, 256, device="cuda", dtype=torch.bfloat16)
        q, scale = convrot_projection_quant(x)
        assert q.shape == x.shape and q.dtype == torch.int8
        assert scale.shape == (17, 1) and scale.dtype == torch.float32
        weight = torch.empty(128, 256, device="cuda", dtype=torch.int8)
        weight_scale = torch.empty(128, 1, device="cuda", dtype=torch.float32)
        out = convrot_projection_mm(q, weight, scale, weight_scale, None, torch.bfloat16)
        assert out.shape == (17, 128) and out.dtype == torch.bfloat16


def _mock_convrot_model(monkeypatch, config):
    events, calls = [], []

    def load(self, weight_dict=None):
        events.append(("weights", weight_dict))

    def invalidate():
        events.append(("invalidate", None))

    def build(modules, packed=False):
        value = SimpleNamespace(modules=modules, packed=packed)
        calls.append(value)
        events.append(("group", value))
        return value

    def wrong_factory(*args, **kwargs):
        pytest.fail("INT8 ConvRot must select its own projection factory")

    monkeypatch.setattr(WanModel, "_apply_weights", load)
    monkeypatch.setattr(ConvRotProjectionGroup, "from_modules", build)
    monkeypatch.setattr(FP8ProjectionGroup, "from_modules", wrong_factory)
    model = EdenLiveModel.__new__(EdenLiveModel)
    model.config = config
    model.transformer_infer = SimpleNamespace(invalidate_cuda_graph=invalidate)
    self_phase = SimpleNamespace(self_attn_q=object(), self_attn_k=object(), self_attn_v=object())
    cross_phase = SimpleNamespace(cross_attn_k=object(), cross_attn_v=object())
    model.transformer_weights = SimpleNamespace(blocks=[SimpleNamespace(compute_phases=[self_phase, cross_phase])])
    return model, self_phase, cross_phase, calls, events


@pytest.mark.parametrize("flags", [{"share_qkv_activation_quant": True}, {"pack_qkv": True}, {"pack_qkv": True, "share_qkv_activation_quant": False}])
def test_int8_model_selects_convrot_groups_and_rebuilds_after_reload(monkeypatch, flags):
    model, self_phase, cross_phase, calls, events = _mock_convrot_model(monkeypatch, {"dit_quant_scheme": "int8-convrot", **flags})
    first_weights = {"version": 1}
    model._apply_weights(first_weights)
    old_self, old_cross = self_phase.self_attn_projection_group, cross_phase.cross_kv_projection_group
    assert old_self.packed is flags.get("pack_qkv", False)
    assert old_self.modules == (self_phase.self_attn_q, self_phase.self_attn_k, self_phase.self_attn_v)
    assert old_cross.packed is False
    assert old_cross.modules == (cross_phase.cross_attn_k, cross_phase.cross_attn_v)
    assert events[:2] == [("invalidate", None), ("weights", first_weights)]
    assert len(calls) == 2
    self_phase.self_attn_q = object()
    cross_phase.cross_attn_k = object()
    second_weights = {"version": 2}
    events.clear()
    model._apply_weights(second_weights)
    assert self_phase.self_attn_projection_group is not old_self
    assert cross_phase.cross_kv_projection_group is not old_cross
    assert self_phase.self_attn_projection_group.modules[0] is self_phase.self_attn_q
    assert cross_phase.cross_kv_projection_group.modules[0] is cross_phase.cross_attn_k
    assert events[:2] == [("invalidate", None), ("weights", second_weights)]
    assert len(calls) == 4


@pytest.mark.parametrize(
    "disabled",
    [
        {"share_qkv_activation_quant": False},
        {"cpu_offload": True},
        {"lazy_load": True},
        {"seq_parallel": True},
        {"tensor_parallel": True},
        {"dit_quant_scheme": "int8-q8f"},
        {"dit_quant_scheme": None},
    ],
)
def test_int8_model_disabled_or_incompatible_configuration_clears_stale_groups(monkeypatch, disabled):
    model, self_phase, cross_phase, calls, events = _mock_convrot_model(monkeypatch, {"share_qkv_activation_quant": True, "dit_quant_scheme": "int8-convrot", **disabled})
    self_phase.self_attn_projection_group = object()
    cross_phase.cross_kv_projection_group = object()
    model._apply_weights()
    assert self_phase.self_attn_projection_group is None
    assert cross_phase.cross_kv_projection_group is None
    assert calls == []
    assert events == [("invalidate", None), ("weights", None)]


def test_int8_model_can_disable_groups_after_initial_load(monkeypatch):
    model, self_phase, cross_phase, calls, _ = _mock_convrot_model(monkeypatch, {"share_qkv_activation_quant": True, "pack_qkv": True, "dit_quant_scheme": "int8-convrot"})
    model._apply_weights()
    assert len(calls) == 2
    model.config.update(share_qkv_activation_quant=False, pack_qkv=False)
    model._apply_weights()
    assert self_phase.self_attn_projection_group is None
    assert cross_phase.cross_kv_projection_group is None
    assert len(calls) == 2


@pytest.mark.parametrize("missing", ["quantize_int8_rowwise_convrot64", "_wrap_for_dlpack", "_C.cutlass_int8_dequant", "registry.get_implementation"])
def test_missing_kitchen_private_api_disables_optimization(monkeypatch, missing):
    if projection_module.kitchen_cuda is None:
        pytest.skip("comfy-kitchen CUDA backend is not installed")
    target, name = projection_module.kitchen_cuda, missing
    if missing.startswith("_C."):
        target, name = target._C, missing.removeprefix("_C.")
    elif missing.startswith("registry."):
        target, name = projection_module.comfy_kitchen.registry, missing.removeprefix("registry.")
        # registry uses slots, so replace the module attribute instead.
        monkeypatch.setattr(projection_module.comfy_kitchen, "registry", SimpleNamespace())
        assert not projection_module._backend_apis_available()
        assert ConvRotProjectionGroup.from_modules(()) is None
        return
    monkeypatch.delattr(target, name)
    assert not projection_module._backend_apis_available()
    assert ConvRotProjectionGroup.from_modules(()) is None


@cuda_only
@pytest.mark.parametrize("bias", [False, True])
def test_packed_group_replaces_original_storage_without_persistent_memory_growth(cuda_state, bias):
    modules = _projections(bias=bias, input_dim=3072, output_dim=3072)
    original_refs = [weakref.ref(module.weight) for module in modules] + [weakref.ref(module.weight_scale) for module in modules]
    if bias:
        original_refs += [weakref.ref(module.bias) for module in modules]
    # CPU reference copies do not retain any of the original CUDA storage.
    expected_weights = tuple(module.weight.cpu() for module in modules)
    expected_scales = tuple(module.weight_scale.cpu() for module in modules)
    expected_biases = tuple(module.bias.cpu() if bias else None for module in modules)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    group = ConvRotProjectionGroup.from_modules(modules, packed=True)
    gc.collect()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert all(reference() is None for reference in original_refs)
    # Repacking replaces the allocations; only allocator rounding can differ.
    assert after - before <= 4096
    weight_storage = group.packed_weight.untyped_storage().data_ptr()
    scale_storage = group.packed_scale.untyped_storage().data_ptr()
    bias_storage = group.packed_bias.untyped_storage().data_ptr() if bias else None
    for index, module in enumerate(modules):
        assert module.weight is group.weights[index]
        assert module.weight_scale is group.scales[index]
        assert module.bias is group.biases[index]
        assert module.weight.is_contiguous() and module.weight_scale.is_contiguous()
        assert module.weight_scale.shape == (3072, 1)
        assert module.weight.untyped_storage().data_ptr() == weight_storage
        assert module.weight_scale.untyped_storage().data_ptr() == scale_storage
        assert torch.equal(module.weight.cpu(), expected_weights[index])
        assert torch.equal(module.weight_scale.cpu(), expected_scales[index])
        if bias:
            assert module.bias.untyped_storage().data_ptr() == bias_storage
            assert torch.equal(module.bias.cpu(), expected_biases[index])


@cuda_only
def test_packed_allocation_failure_leaves_original_modules_untouched(cuda_state, monkeypatch):
    modules = _projections()
    original_weights = tuple(module.weight for module in modules)
    original_scales = tuple(module.weight_scale for module in modules)
    original_biases = tuple(module.bias for module in modules)
    cat = torch.cat
    calls = 0

    def failing_cat(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise torch.OutOfMemoryError("simulated packed bias allocation failure")
        return cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", failing_cat)
    with pytest.raises(torch.OutOfMemoryError):
        ConvRotProjectionGroup.from_modules(modules, packed=True)
    for module, weight, scale, bias in zip(modules, original_weights, original_scales, original_biases):
        assert module.weight is weight
        assert module.weight_scale is scale
        assert module.bias is bias


@cuda_only
def test_packed_groups_can_be_rebuilt_and_keep_original_fallback_exact(cuda_state):
    modules = _projections()
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    expected = tuple(module.apply(x) for module in modules)
    first = ConvRotProjectionGroup.from_modules(modules, packed=True)
    first_weight = weakref.ref(first.packed_weight)
    second = ConvRotProjectionGroup.from_modules(modules, packed=True)
    _assert_equal(expected, first.apply(x))  # Changed module views trigger fallback.
    _assert_equal(expected, second.apply(x))
    _assert_equal(expected, tuple(module.apply(x) for module in modules))
    del first
    gc.collect()
    assert first_weight() is None
