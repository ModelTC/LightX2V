"""Exercise CUTE libraries together: SYCL kernel IDs must not alias across DSOs."""

import itertools
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is unavailable")
@pytest.mark.parametrize("order", list(itertools.permutations(("base", "dense", "sparse"))))
def test_library_load_order(order):
    # A fresh runtime is required for each order; DEVICE_LOST also poisons the
    # process, so never run the native regression in the pytest process itself.
    result = subprocess.run([sys.executable, __file__, *order], capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def run_native_checks(order):
    import sycl_kernels

    root = Path(os.environ.get("CUTE_TEST_LIBRARY_DIR", Path(sycl_kernels.__file__).parent))
    libraries = {
        "base": "cute_fmha_torch.so",
        "dense": "cute_fmha_minimax_h3_torch.so",
        "sparse": "cute_fmha_minimax_h3_sparse_torch.so",
    }
    for name in order:
        torch.ops.load_library(str(root / libraries[name]))

    torch.manual_seed(42)
    q, k, v = [torch.randn((1, 256, 4, 128), device="xpu", dtype=torch.bfloat16) for _ in range(3)]
    lut = torch.arange(2, device="xpu", dtype=torch.int32).view(1, 1, 1, 2).expand(1, 4, 2, 2).contiguous()
    ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    # Launch sparse first, just as the transformer does before VAE decode.
    actual = torch.ops.sycl_kernels_cute_minimax_h3_sparse.sparse_sdp(q, k, v, lut)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, ref, atol=0.01, rtol=0.02)

    for namespace in (torch.ops.sycl_kernels_cute, torch.ops.sycl_kernels_cute_minimax_h3):
        actual = namespace.sdp(q, k, v)
        torch.xpu.synchronize()
        torch.testing.assert_close(actual, ref, atol=0.01, rtol=0.02)

    for sequence in (1792, 1797, 1800):
        q, k, v = [torch.randn((1, sequence, 32, 64), device="xpu", dtype=torch.float16).transpose(1, 2) for _ in range(3)]
        ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
        actual = torch.ops.sycl_kernels_cute.sdp_minimax_h3_vae_d64(q, k, v)
        torch.xpu.synchronize()
        torch.testing.assert_close(actual.float(), ref, atol=0.001, rtol=0.01)
        # Uniform logits must average only valid keys, including a partial KV tile.
        q.zero_()
        k.zero_()
        v.fill_(1)
        actual = torch.ops.sycl_kernels_cute.sdp_minimax_h3_vae_d64(q, k, v)
        torch.xpu.synchronize()
        torch.testing.assert_close(actual, torch.ones_like(actual), atol=0.001, rtol=0)
    print("CUTE load order and attention correctness passed:", order, flush=True)


if __name__ == "__main__":
    run_native_checks(sys.argv[1:])
