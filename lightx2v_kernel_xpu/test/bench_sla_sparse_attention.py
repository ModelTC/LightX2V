#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def measure(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="_cmake_build")
    parser.add_argument("--sequence-length", type=int, default=19292)
    parser.add_argument("--heads", type=int, default=7)
    parser.add_argument("--keep-ratio", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    torch.ops.load_library(str(build_dir / "cute_fmha_minimax_h3_sparse_torch.so"))
    torch.ops.load_library(str(build_dir / "cute_fmha_minimax_h3_torch.so"))
    from sycl_kernels.sla import sla_block_map

    shape = (1, args.sequence_length, args.heads, 128)
    q = torch.randn(shape, device="xpu", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    lut = sla_block_map(q, k, args.keep_ratio, 128, 128)

    sparse_ms = measure(
        lambda: torch.ops.sycl_kernels_cute_minimax_h3_sparse.sparse_sdp(q, k, v, lut),
        args.warmup,
        args.iterations,
    )
    router_ms = measure(
        lambda: sla_block_map(q, k, args.keep_ratio, 128, 128),
        args.warmup,
        args.iterations,
    )
    dense_ms = measure(
        lambda: torch.ops.sycl_kernels_cute_minimax_h3.sdp(q, k, v),
        args.warmup,
        args.iterations,
    )
    sparse_median = statistics.median(sparse_ms)
    router_median = statistics.median(router_ms)
    dense_median = statistics.median(dense_ms)
    print(json.dumps({
        "shape": shape,
        "keep_ratio": args.keep_ratio,
        "lut_shape": list(lut.shape),
        "dense_ms": dense_ms,
        "sparse_kernel_ms": sparse_ms,
        "router_ms": router_ms,
        "kernel_speedup": dense_median / sparse_median,
        "attention_speedup_including_router": dense_median / (sparse_median + router_median),
    }, indent=2))


if __name__ == "__main__":
    main()
