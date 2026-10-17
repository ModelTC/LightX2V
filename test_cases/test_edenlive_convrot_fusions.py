"""Exact BF16 GELU/ConvRot bytes, GEMM output, capture and fallback contracts."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

from lightx2v.models.networks.wan.infer.edenlive import convrot_fusions
from lightx2v.models.networks.wan.infer.edenlive.convrot_fusions import fused_gelu_convrot_int8_quant, fused_gelu_int8_convrot_linear

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="ConvRot requires CUDA")


@pytest.fixture(scope="module")
def precision_extension():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    pytest.importorskip("comfy_kitchen")
    assert convrot_fusions.prepare_convrot_gelu_fusion(), str(convrot_fusions._extension_error)
    return convrot_fusions._extension


def _inputs(tokens, width=14336, outputs=64, bias=True, noncontiguous=False, dtype=torch.bfloat16):
    torch.manual_seed(20261016)
    x = torch.randn(tokens, width * (2 if noncontiguous else 1), device="cuda", dtype=dtype)
    if noncontiguous:
        x = x[:, ::2]
    w = torch.randint(-127, 128, (outputs, width), device="cuda", dtype=torch.int8)
    s = torch.full((outputs, 1), 1 / (127 * math.sqrt(width)), device="cuda", dtype=torch.float32)
    b = torch.randn(outputs, device="cuda", dtype=dtype) * 0.01 if bias else None
    return x, w, s, b


def _reference(x, w, s, b, groupsize=256, out_dtype=torch.bfloat16):
    kitchen = pytest.importorskip("comfy_kitchen")
    return kitchen.int8_linear(F.gelu(x.contiguous(), approximate="tanh"), w, s, b, out_dtype=out_dtype, convrot=True, convrot_groupsize=groupsize)


def _assert_quantized_equal(x):
    kitchen = pytest.importorskip("comfy_kitchen.backends.cuda")
    expected_q, expected_scale = kitchen.quantize_int8_rowwise_convrot64(F.gelu(x.contiguous(), approximate="tanh"), 256)
    q, scale = fused_gelu_convrot_int8_quant(x)
    assert torch.equal(q, expected_q)
    assert torch.equal(scale.view(torch.int32), expected_scale.view(torch.int32))
    assert q.dtype == torch.int8 and q.is_contiguous()
    assert scale.shape == (x.shape[0], 1) and scale.dtype == torch.float32


@pytest.mark.parametrize("tokens,bias", [(880, True), (2640, False)])
def test_fusion_preserves_quant_bytes_scales_and_output_at_streaming_shapes(precision_extension, tokens, bias):
    x, w, s, b = _inputs(tokens, bias=bias)
    original = x.clone()
    _assert_quantized_equal(x)
    actual = fused_gelu_int8_convrot_linear(x, w, s, b, torch.bfloat16, 256)
    assert torch.equal(actual, _reference(x, w, s, b))
    assert actual.shape == (tokens, w.shape[0]) and actual.dtype == torch.bfloat16
    assert torch.equal(x, original)


def test_exact_quantizer_preserves_small_large_and_zero_rows(precision_extension):
    torch.manual_seed(7)
    factors = torch.tensor([0.0, 1e-8, 1e-4, 0.1, 4.0, 32.0], device="cuda", dtype=torch.bfloat16).repeat_interleave(4)
    x = torch.randn(factors.numel(), 14336, device="cuda", dtype=torch.bfloat16) * factors[:, None]
    _assert_quantized_equal(x)


def test_exact_gelu_sweeps_every_finite_bf16_value(precision_extension):
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    x = bits.view(torch.bfloat16)
    x = x[torch.isfinite(x)].cuda()
    actual = torch.empty_like(x)
    error = precision_extension.edenlive_rounded_gelu(x.data_ptr(), actual.data_ptr(), x.numel(), torch.cuda.current_stream().cuda_stream)
    assert error == 0
    eager = F.gelu(x, approximate="tanh")
    compiled = torch.compile(lambda value: F.gelu(value, approximate="tanh"), fullgraph=True, options={"emulate_precision_casts": True})(x)
    assert torch.equal(actual.view(torch.int16), eager.view(torch.int16))
    assert torch.equal(actual.view(torch.int16), compiled.view(torch.int16))


def test_fusion_compiles_fullgraph_for_noncontiguous_input(precision_extension):
    x, w, s, b = _inputs(31, noncontiguous=True)
    compiled = torch.compile(fused_gelu_int8_convrot_linear, fullgraph=True)
    assert torch.equal(compiled(x, w, s, b, torch.bfloat16, 256), _reference(x, w, s, b))


def test_exact_fusion_captures_after_same_stream_warmup(precision_extension):
    x, w, s, b = _inputs(880)
    compiled = torch.compile(fused_gelu_int8_convrot_linear, fullgraph=True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            compiled(x, w, s, b, torch.bfloat16, 256)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        actual = compiled(x, w, s, b, torch.bfloat16, 256)
    x.copy_(torch.randn_like(x))
    graph.replay()
    assert torch.equal(actual, _reference(x, w, s, b))


@cuda_required
@pytest.mark.parametrize(
    "groupsize,width,tokens,dtype",
    [(16, 1024, 33, torch.bfloat16), (64, 1024, 33, torch.bfloat16), (256, 3072, 33, torch.bfloat16), (256, 14336, 1, torch.bfloat16), (256, 14336, 2, torch.bfloat16), (256, 14336, 9, torch.float16)],
)
def test_unsupported_shapes_and_dtypes_fall_back_exactly(groupsize, width, tokens, dtype):
    x, w, s, b = _inputs(tokens, width=width, dtype=dtype)
    expected = _reference(x, w, s, b, groupsize, dtype)
    actual = fused_gelu_int8_convrot_linear(x, w, s, b, dtype, groupsize)
    assert torch.equal(actual, expected)


@cuda_required
def test_missing_compiler_falls_back_to_original_operator(monkeypatch):
    x, w, s, b = _inputs(31)
    monkeypatch.setattr(convrot_fusions, "prepare_convrot_gelu_fusion", lambda device=None: False)
    assert torch.equal(fused_gelu_int8_convrot_linear(x, w, s, b, torch.bfloat16, 256), _reference(x, w, s, b))


@cuda_required
def test_zero_tokens_return_empty_without_native_launch():
    x, w, s, b = _inputs(0)
    actual = fused_gelu_int8_convrot_linear(x, w, s, b, torch.bfloat16, 256)
    assert actual.shape == (0, w.shape[0]) and actual.dtype == torch.bfloat16


def test_fake_output_metadata():
    with FakeTensorMode():
        x = torch.empty((2640, 14336), device="cuda", dtype=torch.bfloat16)
        w = torch.empty((3072, 14336), device="cuda", dtype=torch.int8)
        s = torch.empty((3072, 1), device="cuda", dtype=torch.float32)
        b = torch.empty((3072,), device="cuda", dtype=torch.bfloat16)
        output = fused_gelu_int8_convrot_linear(x, w, s, b, torch.bfloat16, 256)
        assert output.shape == (2640, 3072) and output.dtype == torch.bfloat16
        assert output.device == x.device and output.is_contiguous()
        q, scale = fused_gelu_convrot_int8_quant(x)
        assert q.shape == x.shape and q.dtype == torch.int8
        assert scale.shape == (2640, 1) and scale.dtype == torch.float32


@pytest.mark.parametrize(
    "override,match",
    [
        ({"x_shape": (2, 2, 256)}, "two-dimensional"),
        ({"x_dtype": torch.int8}, "floating input"),
        ({"width": 0}, "nonempty"),
        ({"width": 255}, "divisible"),
        ({"group": 0}, "group size"),
        ({"weight_dtype": torch.bfloat16}, "INT8 weight"),
        ({"scale_dtype": torch.bfloat16}, "FP32 weight scale"),
        ({"out_dtype": torch.int8}, "floating output"),
    ],
)
def test_fake_rejects_unsupported_contracts(override, match):
    with FakeTensorMode():
        width = override.get("width", 256)
        x = torch.empty(override.get("x_shape", (2, width)), device="cuda", dtype=override.get("x_dtype", torch.bfloat16))
        w = torch.empty((64, width), device="cuda", dtype=override.get("weight_dtype", torch.int8))
        s = torch.empty((64, 1), device="cuda", dtype=override.get("scale_dtype", torch.float32))
        with pytest.raises(ValueError, match=match):
            fused_gelu_int8_convrot_linear(x, w, s, None, override.get("out_dtype", torch.bfloat16), override.get("group", 256))
