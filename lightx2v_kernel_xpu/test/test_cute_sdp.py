import math

import pytest
import sycl_kernels
import torch


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cute_sdp_matches_torch(dtype):
    if not sycl_kernels.has_cute_fmha():
        pytest.skip("sycl-kernels was built without CUTE FMHA")

    torch.manual_seed(42)
    q = torch.randn(1, 256, 8, 128, device="xpu", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = sycl_kernels.cute_sdp(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=1.0 / math.sqrt(q.shape[-1]),
    ).transpose(1, 2)

    torch.xpu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
