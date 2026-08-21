import pytest
import sycl_kernels
import torch


def _reference(input_tensor, freqs):
    rotary_dim = freqs.shape[-1]
    half = rotary_dim // 2
    x_rot = input_tensor[..., :rotary_dim]
    x_pass = input_tensor[..., rotary_dim:]
    cos = torch.cos(freqs).to(input_tensor.dtype).unsqueeze(1)
    sin = torch.sin(freqs).to(input_tensor.dtype).unsqueeze(1)
    rotated_half = torch.cat((-x_rot[..., half:], x_rot[..., :half]), dim=-1)
    return torch.cat((x_rot * cos + rotated_half * sin, x_pass), dim=-1)


@pytest.mark.parametrize("shape", [(17, 1, 128), (9, 28, 128), (2, 56, 128)])
def test_minimax_h3_rope_matches_torch(shape):
    if not torch.xpu.is_available():
        pytest.skip("XPU is not available")
    torch.manual_seed(0)
    input_tensor = torch.randn(shape, device="xpu", dtype=torch.bfloat16)
    freqs = torch.randn((shape[0], 96), device="xpu", dtype=torch.float32)
    actual = sycl_kernels.minimax_h3_rope(input_tensor, freqs)
    expected = _reference(input_tensor, freqs)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[..., 96:], input_tensor[..., 96:], atol=0, rtol=0)


def test_minimax_h3_rope_accepts_strided_freqs():
    if not torch.xpu.is_available():
        pytest.skip("XPU is not available")
    input_tensor = torch.randn((3, 2, 128), device="xpu", dtype=torch.bfloat16)
    freqs = torch.randn((3, 104), device="xpu", dtype=torch.float32)[:, :96]
    actual = sycl_kernels.minimax_h3_rope(input_tensor, freqs)
    expected = _reference(input_tensor, freqs)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_minimax_h3_rope_cached_matches_torch():
    if not torch.xpu.is_available():
        pytest.skip("XPU is not available")
    input_tensor = torch.randn((5, 4, 128), device="xpu", dtype=torch.bfloat16)
    freqs = torch.randn((5, 96), device="xpu", dtype=torch.float32)
    actual = sycl_kernels.minimax_h3_rope_cached(input_tensor, freqs.cos(), freqs.sin())
    expected = _reference(input_tensor, freqs)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_minimax_h3_rope_rejects_wrong_rotary_dim():
    if not torch.xpu.is_available():
        pytest.skip("XPU is not available")
    input_tensor = torch.randn((2, 1, 128), device="xpu", dtype=torch.bfloat16)
    freqs = torch.randn((2, 64), device="xpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match=r"\[rows, 96\]"):
        sycl_kernels.minimax_h3_rope(input_tensor, freqs)
