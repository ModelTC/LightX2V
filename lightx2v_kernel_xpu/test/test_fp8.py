"""
Test: onednn_w8a16_fp8
  Supported input dtypes: FP16 (optimised JIT on PTL), BF16 (ref kernel, slower)
  Weight: FP8_E4M3, per-N scale

Shapes tested (LightX2V attention / FFN patterns):
  input [512, 4096]  × weight [4096,  4096]  → attention (to_q / to_k / to_v / to_out)
  input [512, 4096]  × weight [10240, 4096]  → FFN up-proj
  input [512, 10240] × weight [4096,  10240] → FFN down-proj

Tensor shapes:
  x      : [M, K]   FP16 or BF16
  qweight: [N, K]   float8_e4m3fn
  scales : [N, 1]   FP32  (2D; broadcasts over [N, K] for native dequant)

Usage:
  python test_fp8.py            # correctness (FP16+BF16) + bench
  python test_fp8.py --no-bench # correctness only
"""

import sys
import time
import torch
import sycl_kernels

# ── helpers ───────────────────────────────────────────────────────────────────

FP8_MAX = 448.0


def quant_fp8_per_n(weight: torch.Tensor):
    """
    Per-output-channel absmax quantisation → FP8 E4M3.

    weight  : [N, K]  any float dtype
    returns :
      qweight : [N, K]  torch.float8_e4m3fn
      scales  : [N, 1]  torch.float32  (2D for direct broadcast over [N, K])
    """
    assert weight.dim() == 2
    w_f32    = weight.float()
    max_vals = w_f32.abs().max(dim=1).values                        # [N]
    scales   = (max_vals / FP8_MAX).clamp(min=1e-12).unsqueeze(1)  # [N, 1]
    qweight  = (w_f32 / scales).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return qweight, scales  # scales: [N, 1] FP32


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().cpu()
    b_f = b.float().cpu()
    return (
        (a_f - b_f).pow(2).mean().sqrt()
        / (b_f.pow(2).mean().sqrt() + 1e-8)
    ).item()


def dtype_name(dtype: torch.dtype) -> str:
    return {torch.float16: "FP16", torch.bfloat16: "BF16"}.get(dtype, str(dtype))


# ── test shapes ───────────────────────────────────────────────────────────────

SHAPES = [
    (512,  4096,  4096,  "attn  to_q/k/v/out"),
    (512,  10240, 4096,  "FFN   up-proj      "),
    (512,  4096,  10240, "FFN   down-proj    "),
]


# ── correctness ───────────────────────────────────────────────────────────────

def test_correctness(dtype: torch.dtype = torch.float16):
    dname = dtype_name(dtype)
    print("=" * 70)
    print(f"Correctness  ({dname} x FP8 per-N  vs  {dname} x {dname} reference)")
    print("=" * 70)
    all_pass = True

    for M, N, K, label in SHAPES:
        weight      = torch.randn([N, K], dtype=dtype)
        qweight, scales = quant_fp8_per_n(weight)  # scales: [N, 1] FP32

        x           = torch.randn([M, K], dtype=dtype, device="xpu")
        weight_xpu  = weight.to("xpu")
        qweight_xpu = qweight.to("xpu")
        scales_xpu  = scales.to("xpu")              # [N, 1] FP32

        # Reference: same-dtype exact matmul
        ref = torch.nn.functional.linear(x, weight_xpu)

        # OneDNN FP8 kernel  (FP16 → jit:gemm:any;  BF16 → ocl:ref:any on PTL)
        out = sycl_kernels.onednn_w8a16_fp8(x, qweight_xpu, scales_xpu)

        err    = rel_rms(out, ref)
        passed = err < 0.05  # FP8 quantisation typically introduces 2-3% rel RMS
        all_pass &= passed
        print(f"  [{label}]  M={M} N={N:5d} K={K:5d}  "
              f"rel_rms={err:.4f}  {'PASS' if passed else 'FAIL'}")

    print()
    print("All PASS" if all_pass else "*** Some tests FAILED ***")
    print()
    return all_pass


# ── bias correctness ──────────────────────────────────────────────────────────

def test_bias(dtype: torch.dtype = torch.float16):
    dname = dtype_name(dtype)
    print("=" * 70)
    print(f"Bias correctness  ({dname}, attn shape)")
    print("=" * 70)
    M, N, K = 512, 4096, 4096

    weight      = torch.randn([N, K], dtype=dtype)
    qweight, scales = quant_fp8_per_n(weight)
    bias        = torch.randn([N], dtype=dtype)

    x           = torch.randn([M, K], dtype=dtype, device="xpu")
    weight_xpu  = weight.to("xpu")
    qweight_xpu = qweight.to("xpu")
    scales_xpu  = scales.to("xpu")
    bias_xpu    = bias.to("xpu")

    ref = torch.nn.functional.linear(x, weight_xpu, bias_xpu)
    out = sycl_kernels.onednn_w8a16_fp8(x, qweight_xpu, scales_xpu, bias_xpu)

    err    = rel_rms(out, ref)
    passed = err < 0.05  # FP8 quantisation typically introduces 2-3% rel RMS
    print(f"  [attn + bias]  M={M} N={N} K={K}  "
          f"rel_rms={err:.4f}  {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ── benchmark ─────────────────────────────────────────────────────────────────

def bench(M: int, N: int, K: int, label: str,
          dtype: torch.dtype = torch.float16, iters: int = 10):
    dname = dtype_name(dtype)

    weight      = torch.randn([N, K], dtype=dtype)
    qweight, scales = quant_fp8_per_n(weight)

    x           = torch.randn([M, K], dtype=dtype, device="xpu")
    qweight_xpu = qweight.to("xpu")   # [N, K]  float8_e4m3fn
    scales_xpu  = scales.to("xpu")    # [N, 1]  FP32

    # warmup both paths
    for _ in range(10):
        sycl_kernels.onednn_w8a16_fp8(x, qweight_xpu, scales_xpu)
    for _ in range(10):
        torch.nn.functional.linear(
            x, qweight_xpu.to(dtype) * scales_xpu.to(dtype))
    torch.xpu.synchronize()

    # ── onednn fp8 ────────────────────────────────────────────────────────────
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        output_onednn = sycl_kernels.onednn_w8a16_fp8(x, qweight_xpu, scales_xpu)
    torch.xpu.synchronize()
    elapsed_onednn = time.perf_counter() - t0
    ms_on     = elapsed_onednn / iters * 1e3
    tflops_on = 2 * M * N * K / (elapsed_onednn / iters) / 1e12
    print(f"  onednn_{dname.lower()}_fp8  [{label}]  "
          f"{ms_on:.3f} ms/iter  {tflops_on:.2f} TFLOPS")

    # ── native: dequant (fp8 → dtype) + dtype gemm ───────────────────────────
    # scales_xpu [N,1] * qweight [N,K] → [N,K] correct per-row broadcast
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        w_dq          = qweight_xpu.to(dtype) * scales_xpu.to(dtype)  # [N,K]
        output_native = torch.nn.functional.linear(x, w_dq)
    torch.xpu.synchronize()
    elapsed_native = time.perf_counter() - t0
    ms_na     = elapsed_native / iters * 1e3
    tflops_na = 2 * M * N * K / (elapsed_native / iters) / 1e12
    print(f"  native_{dname.lower()}       [{label}]  "
          f"{ms_na:.3f} ms/iter  {tflops_na:.2f} TFLOPS")

    # ── compare outputs & speedup ─────────────────────────────────────────────
    err     = rel_rms(output_onednn, output_native)
    speedup = elapsed_native / elapsed_onednn
    print(f"  rel_rms(onednn vs native)={err:.4f}  "
          f"speedup={speedup:.2f}x  ({'onednn faster' if speedup > 1 else 'native faster'})")
    print()


def run_bench():
    for dtype in [torch.float16, torch.bfloat16]:
        dname = dtype_name(dtype)
        print("=" * 70)
        print(f"Performance benchmark  {dname} x FP8  "
              f"({'jit:gemm' if dtype == torch.float16 else 'ocl:ref — slow on PTL'})")
        print("=" * 70)
        for M, N, K, label in SHAPES:
            bench(M, N, K, label, dtype=dtype)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    no_bench = "--no-bench" in sys.argv

    ok  = test_correctness(torch.float16)
    ok &= test_correctness(torch.bfloat16)
    ok &= test_bias(torch.float16)
    ok &= test_bias(torch.bfloat16)

    if not no_bench:
        run_bench()

    sys.exit(0 if ok else 1)
