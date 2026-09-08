#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch


def synchronize():
    torch.xpu.synchronize()


def measure(op, q, k, v, iterations):
    samples = []
    for _ in range(iterations):
        synchronize()
        start = time.perf_counter()
        output = op(q, k, v)
        synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return output, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="lightx2v_kernel_xpu/_cmake_build")
    parser.add_argument("--sequence-length", type=int, default=19292)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimized-first", action="store_true")
    parser.add_argument("--variant", choices=("both", "generic", "optimized"), default="both")
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    if args.variant in ("both", "generic"):
        torch.ops.load_library(str(build_dir / "cute_fmha_torch.so"))
    if args.variant in ("both", "optimized"):
        torch.ops.load_library(str(build_dir / "cute_fmha_minimax_h3_torch.so"))
    generator = torch.Generator(device="xpu").manual_seed(args.seed)
    shape = (1, args.sequence_length, args.heads, 128)
    q = torch.randn(shape, device="xpu", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(shape, device="xpu", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(shape, device="xpu", dtype=torch.bfloat16, generator=generator)

    if args.variant == "generic":
        _, samples = measure(torch.ops.sycl_kernels_cute.sdp, q, k, v, args.iterations)
        print(json.dumps({"variant": "generic", "shape": shape, "milliseconds": samples}, indent=2))
        return
    if args.variant == "optimized":
        _, samples = measure(torch.ops.sycl_kernels_cute_minimax_h3.sdp, q, k, v, args.iterations)
        print(json.dumps({"variant": "optimized", "shape": shape, "milliseconds": samples}, indent=2))
        return
    if args.optimized_first:
        optimized, optimized_ms = measure(torch.ops.sycl_kernels_cute_minimax_h3.sdp, q, k, v, args.iterations)
        generic, generic_ms = measure(torch.ops.sycl_kernels_cute.sdp, q, k, v, args.iterations)
    else:
        generic, generic_ms = measure(torch.ops.sycl_kernels_cute.sdp, q, k, v, args.iterations)
        optimized, optimized_ms = measure(torch.ops.sycl_kernels_cute_minimax_h3.sdp, q, k, v, args.iterations)
    error = (generic.float() - optimized.float()).abs()
    result = {
        "shape": shape,
        "dtype": "bfloat16",
        "iterations": args.iterations,
        "order": "optimized,generic" if args.optimized_first else "generic,optimized",
        "generic_ms": generic_ms,
        "optimized_ms": optimized_ms,
        "speedup": (sum(generic_ms) / sum(optimized_ms)),
        "max_abs_error": error.max().item(),
        "mean_abs_error": error.mean().item(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
