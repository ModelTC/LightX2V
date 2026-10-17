"""Packed FP8 projections: numerical parity, adapter safety, and strided KV."""

from types import SimpleNamespace

import pytest
import torch

import lightx2v.common.ops.mm.mm_weight as mm_weight_module
import lightx2v.models.networks.wan.infer.edenlive.packed_projection as projection_module
from lightx2v.common.ops.attn.sage_sm90_fused import make_transposed_v, update_transposed_v
from lightx2v.common.ops.mm.mm_weight import MMWeightWfp8channelAfp8channeldynamicSgl
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightSgl
from lightx2v.models.networks.wan.edenlive_model import EdenLiveModel
from lightx2v.models.networks.wan.infer.edenlive.packed_projection import FP8ProjectionGroup
from lightx2v.models.networks.wan.infer.edenlive.transformer_infer import EdenLiveTransformerInfer
from lightx2v.models.networks.wan.model import WanModel

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8-SGL and transposed V require CUDA")


@pytest.fixture
def cuda_state():
    pytest.importorskip("sgl_kernel")
    torch.manual_seed(123)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _projections(bias=True, dtype=torch.bfloat16):
    modules = []
    for name in ("q", "k", "v"):
        module = MMWeightWfp8channelAfp8channeldynamicSgl(name + ".weight", name + ".bias" if bias else None)
        module.infer_dtype = dtype
        module.weight = torch.randn(128, 256, device="cuda", dtype=dtype).to(torch.float8_e4m3fn).t()
        module.weight_scale = torch.rand(128, 1, device="cuda", dtype=torch.float32).mul_(0.02).add_(0.01)
        module.bias = torch.randn(128, device="cuda", dtype=dtype) if bias else None
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
def test_projection_group_eager_and_fullgraph_preserve_values_and_inputs(cuda_state, packed, bias, dtype):
    modules = _projections(bias=bias, dtype=dtype)
    original_weights = tuple(module.weight.t().view(torch.uint8).clone() for module in modules)
    group = FP8ProjectionGroup.from_modules(modules, packed=packed)
    assert group is not None
    compiled = torch.compile(group.apply, fullgraph=True)
    with torch.inference_mode():
        for tokens in (17, 67):
            x = torch.randn(tokens, 256, device="cuda", dtype=dtype)
            original_x = x.clone()
            expected = tuple(module.apply(x) for module in modules)
            for apply in (group.apply, compiled):
                actual = apply(x)
                _assert_equal(expected, actual)
                assert all(output.stride() == ((384 if packed else 128), 1) for output in actual)
                assert torch.equal(x, original_x)
    for module, original in zip(modules, original_weights):
        assert torch.equal(module.weight.t().view(torch.uint8), original)


@cuda_only
def test_quantization_and_gemm_counts_drop_without_changing_values(cuda_state, monkeypatch):
    modules = _projections()
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    original_quant = projection_module.sgl_fp8_per_token_quant
    original_mm = projection_module.sgl_fp8_scaled_mm
    counts = {"quant": 0, "mm": 0}

    def quantize(value):
        counts["quant"] += 1
        return original_quant(value)

    def mm(*args, **kwargs):
        counts["mm"] += 1
        return original_mm(*args, **kwargs)

    monkeypatch.setattr(mm_weight_module, "magi_register_custom_op", None)
    for namespace in (projection_module, mm_weight_module):
        monkeypatch.setattr(namespace, "sgl_fp8_per_token_quant", quantize)
        monkeypatch.setattr(namespace, "sgl_fp8_scaled_mm", mm)
    expected = tuple(module.apply(x) for module in modules)
    assert counts == {"quant": 3, "mm": 3}
    for packed, expected_mm in ((False, 3), (True, 1)):
        counts.update(quant=0, mm=0)
        group = FP8ProjectionGroup.from_modules(modules, packed=packed)
        _assert_equal(expected, group.apply(x))
        assert counts == {"quant": 1, "mm": expected_mm}


@cuda_only
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("change", ["lora", "diff", "weight", "scale", "bias"])
def test_dynamic_adapters_and_tensor_replacements_fall_back_after_compilation(cuda_state, packed, change):
    modules = _projections()
    group = FP8ProjectionGroup.from_modules(modules, packed=packed)
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(group.apply, fullgraph=True)
    before = compiled(x)
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
        module.weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
    elif change == "scale":
        module.weight_scale = module.weight_scale * 2
    else:
        module.bias = module.bias + 1
    expected = tuple(projection.apply(x) for projection in modules)
    assert not torch.equal(before[0], expected[0])
    _assert_equal(expected, group.apply(x))
    _assert_equal(expected, compiled(x))


@cuda_only
@pytest.mark.parametrize("change", ["lora", "diff", "weight", "scale", "bias"])
def test_stale_packed_group_calls_each_original_projection(cuda_state, monkeypatch, change):
    modules = _projections()
    group = FP8ProjectionGroup.from_modules(modules, packed=True)
    module = modules[0]
    if change == "lora":
        module.has_lora_branch = True
    elif change == "diff":
        module.has_diff = True
    elif change == "weight":
        module.weight = module.weight.clone()
    elif change == "scale":
        module.weight_scale = module.weight_scale.clone()
    else:
        module.bias = module.bias.clone()
    calls = []
    sentinels = tuple(object() for _ in modules)
    for index, projection in enumerate(modules):

        def apply(x, index=index):
            calls.append(index)
            return sentinels[index]

        monkeypatch.setattr(projection, "apply", apply)
    actual = group.apply(torch.zeros(17, 256, device="cuda", dtype=torch.bfloat16))
    assert actual == sentinels
    assert calls == [0, 1, 2]


def test_empty_wrong_type_and_cpu_groups_are_rejected():
    assert FP8ProjectionGroup.from_modules(()) is None
    assert FP8ProjectionGroup.from_modules((object(),)) is None
    module = MMWeightWfp8channelAfp8channeldynamicSgl("weight", None)
    module.weight = torch.empty(128, 256, dtype=torch.float8_e4m3fn).t()
    assert FP8ProjectionGroup.from_modules((module,)) is None


@cuda_only
@pytest.mark.parametrize("incompatibility", ["lora", "diff", "lazy_load", "create_cuda_buffer", "create_cpu_buffer", "mixed_bias", "wrong_scale", "wrong_dtype", "row_major"])
def test_incompatible_resident_groups_are_rejected(cuda_state, incompatibility):
    modules = _projections()
    module = modules[-1]
    if incompatibility == "lora":
        module.has_lora_branch = True
    elif incompatibility == "diff":
        module.has_diff = True
    elif incompatibility in ("lazy_load", "create_cuda_buffer", "create_cpu_buffer"):
        setattr(module, incompatibility, True)
    elif incompatibility == "mixed_bias":
        module.bias = None
    elif incompatibility == "wrong_scale":
        module.weight_scale = module.weight_scale[:1]
    elif incompatibility == "wrong_dtype":
        module.infer_dtype = torch.float32
    else:
        module.weight = module.weight.contiguous()
    assert FP8ProjectionGroup.from_modules(modules, packed=True) is None


@cuda_only
@pytest.mark.parametrize("offset", [0, 3072])
def test_compiled_projection_norm_handles_packed_qk_stride_exactly(cuda_state, offset):
    pytest.importorskip("flashinfer.norm")
    norm = RMSWeightSgl(None)
    norm.infer_dtype = norm.sensitive_layer_dtype = torch.bfloat16
    norm.weight = torch.randn(3072, device="cuda", dtype=torch.bfloat16)
    packed = torch.randn(67, 9216, device="cuda", dtype=torch.bfloat16)
    projected = packed[:, offset : offset + 3072]
    assert projected.stride() == (9216, 1) and not projected.is_contiguous()
    ordinary = torch.compile(norm.apply, fullgraph=True)
    optimized = torch.compile(lambda value: EdenLiveTransformerInfer._projection_norm(norm, value), fullgraph=True)
    with torch.inference_mode():
        expected = ordinary(projected)
        actual = optimized(projected)
    assert torch.equal(actual, expected)
    assert actual.view(-1, 24, 128).shape == (67, 24, 128)


@cuda_only
@pytest.mark.parametrize("layout", ["packed", "stepped"])
@pytest.mark.parametrize("length,start", [(0, 7), (1, 3), (17, 19), (35, 7)])
def test_transposed_v_updates_accept_strides_offsets_and_partial_tiles(cuda_state, layout, length, start):
    batch, heads, dim = 2, 3, 65
    if layout == "packed":
        values = torch.randn(batch, length + 3, 3 * heads * dim, device="cuda", dtype=torch.bfloat16)
        v = values[:, 1 : length + 1, 2 * heads * dim :].view(batch, length, heads, dim)
    else:
        values = torch.randn(batch, length + 3, heads, 2 * dim, device="cuda", dtype=torch.bfloat16)
        v = values[:, 1 : length + 1, :, ::2]
    assert v.storage_offset() > 0
    workspace = torch.full((batch, dim, heads, 128), 23, device="cuda", dtype=torch.bfloat16)
    expected = workspace.clone()
    contiguous = workspace.clone()
    positions = torch.arange(start, start + length, device="cuda")
    permuted = (positions & ~15) | (positions & 1) | ((positions & 8) >> 2) | ((positions & 2) << 1) | ((positions & 4) << 1)
    expected[:, :, :, permuted] = v.permute(0, 3, 2, 1)
    update_transposed_v(v, workspace, start)
    update_transposed_v(v.contiguous(), contiguous, start)
    assert torch.equal(workspace, expected)
    assert torch.equal(workspace, contiguous)
    initial = make_transposed_v(v, 127)
    assert torch.equal(initial, make_transposed_v(v.contiguous(), 127))


def _mock_model(monkeypatch, config):
    monkeypatch.setattr(WanModel, "_apply_weights", lambda self, weight_dict=None: None)
    model = EdenLiveModel.__new__(EdenLiveModel)
    model.config = config
    self_phase = SimpleNamespace(self_attn_q=object(), self_attn_k=object(), self_attn_v=object())
    cross_phase = SimpleNamespace(cross_attn_k=object(), cross_attn_v=object())
    model.transformer_weights = SimpleNamespace(blocks=[SimpleNamespace(compute_phases=[self_phase, cross_phase])])
    calls = []

    def build(modules, packed=False):
        value = SimpleNamespace(modules=modules, packed=packed)
        calls.append(value)
        return value

    monkeypatch.setattr(FP8ProjectionGroup, "from_modules", build)
    return model, self_phase, cross_phase, calls


@pytest.mark.parametrize("flag", ["pack_qkv", "share_qkv_activation_quant"])
def test_resident_single_card_model_builds_groups_and_rebuilds_after_reload(monkeypatch, flag):
    model, self_phase, cross_phase, calls = _mock_model(monkeypatch, {flag: True, "dit_quant_scheme": "fp8-sgl"})
    model._apply_weights()
    old = self_phase.self_attn_projection_group
    assert old.packed is (flag == "pack_qkv")
    assert cross_phase.cross_kv_projection_group.packed is False
    assert len(calls) == 2
    self_phase.self_attn_q = object()
    model._apply_weights()
    assert self_phase.self_attn_projection_group is not old
    assert self_phase.self_attn_projection_group.modules[0] is self_phase.self_attn_q
    assert len(calls) == 4


@pytest.mark.parametrize("disabled", [{"cpu_offload": True}, {"lazy_load": True}, {"seq_parallel": True}, {"tensor_parallel": True}, {"dit_quant_scheme": "fp8-f16-accum"}, {"pack_qkv": False}])
def test_model_compatibility_gate_clears_groups_without_touching_parallel_paths(monkeypatch, disabled):
    config = {"pack_qkv": True, "dit_quant_scheme": "fp8-sgl"}
    config.update(disabled)
    model, self_phase, cross_phase, calls = _mock_model(monkeypatch, config)
    self_phase.self_attn_projection_group = object()
    cross_phase.cross_kv_projection_group = object()
    model._apply_weights()
    assert self_phase.self_attn_projection_group is None
    assert cross_phase.cross_kv_projection_group is None
    assert calls == []
