import pytest
import sycl_kernels
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(17, 128), (9, 5376), (2, 8192)])
def test_rms_norm_matches_torch(dtype, shape):
    if not torch.xpu.is_available():
        pytest.skip("XPU is not available")
    torch.manual_seed(0)
    x = torch.randn(shape, device="xpu", dtype=dtype)
    weight = torch.randn(shape[-1], device="xpu", dtype=dtype)
    actual = sycl_kernels.rms_norm(weight, x, 1e-6)
    expected = torch.nn.functional.rms_norm(x.float(), (shape[-1],), weight=weight.float(), eps=1e-6).to(dtype)
    torch.xpu.synchronize()
    atol = 2e-2 if dtype != torch.float32 else 2e-5
    rtol = 2e-2 if dtype != torch.float32 else 2e-5
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_rms_norm_rejects_unsupported_hidden_size():
    if not torch.xpu.is_available():
        pytest.skip("XPU is not available")
    x = torch.randn((2, 127), device="xpu", dtype=torch.bfloat16)
    weight = torch.ones(127, device="xpu", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="divisible by 32"):
        sycl_kernels.rms_norm(weight, x, 1e-6)
