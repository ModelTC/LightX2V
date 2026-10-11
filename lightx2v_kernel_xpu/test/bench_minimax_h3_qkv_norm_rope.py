"""Kernel-only microbenchmark for MiniMax-H3 QKV norm + RoPE fusion."""

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lightx2v_kernel_xpu/python"))


def load_module(relative_path: str):
    path = ROOT / "lightx2v/models/networks/minimax_h3" / relative_path
    name = "lightx2v.models.networks.minimax_h3." + relative_path.removesuffix(".py").replace("/", ".")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def percentile(samples, fraction):
    values = sorted(samples)
    return values[round((len(values) - 1) * fraction)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("xpu", "cuda"), default="xpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=(1024, 4096, 19200))
    parser.add_argument("--heads", type=int, nargs="+", default=(7,))
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--fused-backend", choices=("triton", "intel_xpu"), default="triton")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    device_api = getattr(torch, args.device)
    if not device_api.is_available():
        raise SystemExit(f"{args.device.upper()} is unavailable")
    device_api.set_device(args.device_index)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    qkv_ops = load_module("infer/fused_qkv.py")
    if args.device != "xpu" or dtype != torch.bfloat16:
        raise SystemExit("The Intel XPU RoPE baseline requires --device xpu --dtype bf16")
    import sycl_kernels

    if not sycl_kernels.has_minimax_h3_rope():
        raise SystemExit("The Intel XPU MiniMax-H3 RoPE extension is unavailable")
    if args.fused_backend == "intel_xpu" and not sycl_kernels.has_minimax_h3_qkv_norm_rope():
        raise SystemExit("The Intel XPU MiniMax-H3 fused QKV norm + RoPE extension is unavailable")
    device = torch.device(args.device, args.device_index)
    dim, rotary = 128, 96

    print(f"device={device} ({device_api.get_device_name(args.device_index)}) dtype={args.dtype} dim={dim} rotary={rotary}")
    print("tokens heads  separate_ms  fused_ms  speedup  reduction")
    for tokens in args.tokens:
        for heads in args.heads:
            packed = torch.randn(tokens, 3 * heads * dim, device=device, dtype=dtype)
            q, k, v = packed.chunk(3, -1)
            qw = torch.randn(dim, device=device, dtype=dtype)
            kw = torch.randn(dim, device=device, dtype=dtype)
            phases = torch.randn(tokens, rotary, device=device, dtype=torch.float32)
            cos, sin = phases.cos(), phases.sin()

            def separate(packed=packed, heads=heads, qw=qw, kw=kw, cos=cos, sin=sin):
                q, k, v = (part.unflatten(-1, (heads, dim)) for part in packed.chunk(3, -1))
                q = F.rms_norm(q.float(), (dim,), qw.float(), 1e-5).to(dtype)
                k = F.rms_norm(k.float(), (dim,), kw.float(), 1e-5).to(dtype)
                return sycl_kernels.minimax_h3_rope_cached(q, cos, sin), sycl_kernels.minimax_h3_rope_cached(k, cos, sin), v

            def fused(q=q, k=k, v=v, qw=qw, kw=kw, cos=cos, sin=sin):
                if args.fused_backend == "intel_xpu":
                    return sycl_kernels.minimax_h3_qkv_norm_rope(q, k, v, qw, kw, cos, sin, 1e-5, 1e-5, True)
                return qkv_ops.split_qkv_norm_rope(q, k, v, qw, kw, cos, sin, 1e-5, 1e-5)

            # Compile both variants and validate that the benchmarked kernels agree.
            separate_outputs = separate()
            fused_outputs = fused()
            device_api.synchronize()
            for index, (actual, expected) in enumerate(zip(fused_outputs, separate_outputs)):
                if index == 2:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                else:
                    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=3e-2)
            for _ in range(args.warmup):
                separate()
                fused()
            device_api.synchronize()

            timings = {"separate": [], "fused": []}
            for sample in range(args.samples):
                # Alternate order to reduce clock/thermal ordering bias.
                variants = (("separate", separate), ("fused", fused))
                if sample % 2:
                    variants = variants[::-1]
                for name, function in variants:
                    start = device_api.Event(enable_timing=True)
                    end = device_api.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        function()
                    end.record()
                    end.synchronize()
                    timings[name].append(start.elapsed_time(end) / args.iterations)

            separate_ms = statistics.median(timings["separate"])
            fused_ms = statistics.median(timings["fused"])
            print(f"{tokens:6d} {heads:5d} {separate_ms:12.4f} {fused_ms:9.4f} {separate_ms / fused_ms:7.3f}x {100 * (1 - fused_ms / separate_ms):8.2f}%")
            print(
                " " * 13
                + f"p10/p90 separate={percentile(timings['separate'], 0.1):.4f}/{percentile(timings['separate'], 0.9):.4f} ms "
                + f"fused={percentile(timings['fused'], 0.1):.4f}/{percentile(timings['fused'], 0.9):.4f} ms"
            )


if __name__ == "__main__":
    main()
