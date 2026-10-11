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


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is unavailable")
@pytest.mark.parametrize("sequence", [261, 1797])
@pytest.mark.parametrize("packed_v", [False, True])
def test_minimax_h3_vae_sdp_d64_matches_torch(sequence, packed_v):
    if not sycl_kernels.has_minimax_h3_vae_sdp_d64():
        pytest.skip("sycl-kernels was built without MiniMax-H3 VAE D64 CUTE FMHA")

    torch.manual_seed(42)
    q = torch.randn(1, sequence, 32, 64, device="xpu", dtype=torch.float16).transpose(1, 2)
    k = torch.randn_like(q)
    if packed_v:
        qkv = torch.randn(1, sequence, 32, 192, device="xpu", dtype=torch.float16)
        v = qkv[..., 128:].transpose(1, 2)
    else:
        v = torch.randn_like(q)

    actual = sycl_kernels.minimax_h3_vae_sdp_d64(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    torch.xpu.synchronize()
    assert actual.stride() == q.stride()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=8e-3)
