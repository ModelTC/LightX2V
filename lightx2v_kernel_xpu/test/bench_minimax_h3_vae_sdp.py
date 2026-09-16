#!/usr/bin/env python3
import argparse
import time

import sycl_kernels
import torch


def measure(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=1797)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(42)
    shape = (1, args.sequence, 32, 64)
    q = torch.randn(shape, device="xpu", dtype=torch.float16).transpose(1, 2)
    k = torch.randn(shape, device="xpu", dtype=torch.float16).transpose(1, 2)
    v_contiguous = torch.randn(shape, device="xpu", dtype=torch.float16).transpose(1, 2)
    qkv = torch.randn((1, args.sequence, 32, 192), device="xpu", dtype=torch.float16)
    v_packed = qkv[..., 128:].transpose(1, 2)

    cases = {
        "torch_sdpa_contiguous": lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v_contiguous),
        "cute_contiguous": lambda: sycl_kernels.minimax_h3_vae_sdp_d64(q, k, v_contiguous),
        "torch_sdpa_packed": lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v_packed),
        "cute_packed": lambda: sycl_kernels.minimax_h3_vae_sdp_d64(q, k, v_packed),
    }
    results = {name: measure(fn, args.warmup, args.iterations) for name, fn in cases.items()}
    for name, elapsed_ms in results.items():
        print(f"{name}: {elapsed_ms:.4f} ms")
    print(f"CUTE contiguous speedup: {results['torch_sdpa_contiguous'] / results['cute_contiguous']:.3f}x")
    print(f"CUTE packed speedup: {results['torch_sdpa_packed'] / results['cute_packed']:.3f}x")


if __name__ == "__main__":
    main()
