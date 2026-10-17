"""Strict FP8-SGL FFN activation parity and compiled custom-op metadata."""

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

from lightx2v.models.networks.wan.infer.edenlive.fp8_fusions import fused_gelu_quant_fp8_sgl


@pytest.fixture(scope="module")
def compiled_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    sgl_kernel = pytest.importorskip("sgl_kernel")

    @torch.library.custom_op("edenlive_ffn_test::sgl_quant", mutates_args=(), device_types="cuda")
    def quant(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
        scales = input_tensor.new_empty((input_tensor.shape[0], 1), dtype=torch.float32)
        sgl_kernel.sgl_per_token_quant_fp8(input_tensor, output, scales)
        return output, scales

    @quant.register_fake
    def quant_fake(input_tensor):
        return input_tensor.new_empty(input_tensor.shape, dtype=torch.float8_e4m3fn), input_tensor.new_empty((input_tensor.shape[0], 1), dtype=torch.float32)

    def reference(input_tensor):
        return quant(F.gelu(input_tensor, approximate="tanh"))

    return torch.compile(reference, fullgraph=True, options={"emulate_precision_casts": True})


def _assert_equal(actual, expected):
    quantized, scales = actual
    expected_quantized, expected_scales = expected
    assert quantized.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
    assert quantized.shape == expected_quantized.shape and scales.shape == (quantized.shape[0], 1)
    assert quantized.is_contiguous() and scales.is_contiguous()
    # Compare FP8 storage too, so signed zero and exact rounding are covered.
    assert torch.equal(quantized.view(torch.uint8), expected_quantized.view(torch.uint8))
    assert torch.equal(scales.view(torch.int32), expected_scales.view(torch.int32))


@pytest.mark.parametrize("tokens", [880, 2640])
def test_fused_ffn_matches_compiled_reference_at_streaming_shapes(compiled_reference, tokens):
    torch.manual_seed(20261016)
    x = torch.randn(tokens, 14336, device="cuda", dtype=torch.bfloat16)
    original = x.clone()
    _assert_equal(fused_gelu_quant_fp8_sgl(x), compiled_reference(x))
    assert torch.equal(x, original)


def test_fused_ffn_preserves_small_wide_and_zero_rows(compiled_reference):
    torch.manual_seed(42)
    factors = torch.tensor([0.0, 1e-8, 1e-4, 0.1, 4.0, 32.0], device="cuda", dtype=torch.bfloat16).repeat_interleave(4)
    x = torch.randn(factors.numel(), 14336, device="cuda", dtype=torch.bfloat16) * factors[:, None]
    actual = fused_gelu_quant_fp8_sgl(x)
    _assert_equal(actual, compiled_reference(x))
    assert torch.count_nonzero(actual[1][:4]) == 0
    assert torch.all(actual[0][:4].float() == 448)


def test_fused_ffn_sweeps_every_finite_bf16_value(compiled_reference):
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    total = 5 * 14336
    x = values.repeat((total + values.numel() - 1) // values.numel())[:total].reshape(5, 14336).to("cuda")
    _assert_equal(fused_gelu_quant_fp8_sgl(x), compiled_reference(x))


def test_fused_ffn_compiles_fullgraph_with_noncontiguous_input(compiled_reference):
    torch.manual_seed(7)
    x = torch.randn(9, 14336 * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    compiled = torch.compile(fused_gelu_quant_fp8_sgl, fullgraph=True)
    _assert_equal(compiled(x), compiled_reference(x.contiguous()))


def test_fake_metadata_and_empty_token_dimension():
    with FakeTensorMode():
        x = torch.empty(0, 14336, device="cuda", dtype=torch.bfloat16)
        quantized, scales = fused_gelu_quant_fp8_sgl(x)
        assert quantized.shape == (0, 14336) and quantized.dtype == torch.float8_e4m3fn
        assert scales.shape == (0, 1) and scales.dtype == torch.float32
        assert quantized.device == x.device and scales.device == x.device


@pytest.mark.parametrize("shape,dtype", [((4, 8), torch.float16), ((4, 8, 2), torch.bfloat16), ((4, 0), torch.bfloat16)])
def test_fake_rejects_unsupported_input(shape, dtype):
    with FakeTensorMode():
        x = torch.empty(shape, device="cuda", dtype=dtype)
        with pytest.raises(ValueError, match="requires"):
            fused_gelu_quant_fp8_sgl(x)
