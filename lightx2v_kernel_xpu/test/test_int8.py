import pytest
import sycl_kernels
import torch


def quantize_int8_per_output_channel(weight):
    weight_f32 = weight.float()
    scales = (weight_f32.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    qweight = torch.round(weight_f32 / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return qweight, scales


def relative_rms(actual, expected):
    actual = actual.float()
    expected = expected.float()
    return ((actual - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)).item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_onednn_w8a8_int8(dtype, with_bias):
    if not torch.xpu.is_available():
        pytest.skip("XPU is not available")

    torch.manual_seed(1234)
    M, N, K = 64, 192, 256
    weight = torch.randn(N, K, dtype=dtype)
    qweight, scales = quantize_int8_per_output_channel(weight)

    x = torch.randn(M, K, dtype=dtype, device="xpu")
    qweight = qweight.to("xpu")
    scales = scales.to("xpu")
    bias = torch.randn(N, dtype=dtype, device="xpu") if with_bias else None

    x_f32 = x.float()
    x_scales = (x_f32.abs().amax(dim=1) / 127.0).clamp_min(1e-30)
    qx = torch.round(x_f32 / x_scales[:, None]).clamp(-127, 127)
    dequantized_x = qx.to(dtype) * x_scales.to(dtype)[:, None]
    dequantized_weight = qweight.to(dtype) * scales.to(dtype)[:, None]
    expected = torch.nn.functional.linear(dequantized_x, dequantized_weight, bias)
    actual = sycl_kernels.onednn_w8a8_int8(x, qweight, scales, bias)
    torch.xpu.synchronize()

    assert relative_rms(actual, expected) < 0.02


def test_onednn_w8a8_int8_rejects_wrong_weight_dtype():
    if not torch.xpu.is_available():
        pytest.skip("XPU is not available")

    x = torch.randn(2, 16, dtype=torch.bfloat16, device="xpu")
    weight = torch.randn(4, 16, dtype=torch.bfloat16, device="xpu")
    scales = torch.ones(4, dtype=torch.float32, device="xpu")
    with pytest.raises(RuntimeError, match="torch.int8"):
        sycl_kernels.onednn_w8a8_int8(x, weight, scales)
