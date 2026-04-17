"""
SDP Flash Attention test for PTL-H ESIMD kernels.

Test shapes from LightX2V WAN inference:
  Self-attn:  Q=[14040, 12, 128], K=[14040, 12, 128], V=[14040, 12, 128]
  Cross-attn: Q=[14040, 12, 128], K=[512,   12, 128], V=[512,   12, 128]

Our kernel API: sycl_kernels.sdp(Q, K, V)
  [B, L, H, D] with B=1, D=128
  dtype: fp16 -> sdp_fp16 kernel, bf16 -> sdp_bf16io kernel (auto-dispatched)
"""

import math
import time

import sycl_kernels
import torch

DEVICE = "xpu"
HD = 128


# ──────────────────────────────────────────────────────────────────────────────
# CPU reference
# ──────────────────────────────────────────────────────────────────────────────
def cpu_sdp_ref(Q_f32, K_f32, V_f32, scale):
    q = Q_f32.permute(1, 0, 2)
    k = K_f32.permute(1, 0, 2)
    v = V_f32.permute(1, 0, 2)
    scores = torch.bmm(q, k.transpose(1, 2)) * scale
    weights = torch.softmax(scores.float(), dim=-1)
    out = torch.bmm(weights, v)
    return out.permute(1, 0, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Correctness check
# ──────────────────────────────────────────────────────────────────────────────
def check(name, out_f32, ref_f32, thresh=0.15):
    diff = (out_f32 - ref_f32).abs()
    max_d = diff.max().item()
    rms = (diff.pow(2).sum() / ref_f32.pow(2).sum()).sqrt().item()
    nan_c = out_f32.isnan().sum().item()
    ok = (nan_c == 0) and (max_d < thresh)
    status = "PASS" if ok else "FAIL"
    print(f"  {name}: {status}  max_diff={max_d:.4f}  rel_rms={rms:.2e}  NaN={nan_c}")
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# One test case
# ──────────────────────────────────────────────────────────────────────────────
def run_test(label, q_len, kv_len, num_heads=12, warmup=10, iters=50):
    print(f"\n{'=' * 60}")
    print(f"  {label}: Q=[{q_len},{num_heads},{HD}] K/V=[{kv_len},{num_heads},{HD}]")
    print(f"{'=' * 60}")

    scale = 1.0 / math.sqrt(HD)
    torch.manual_seed(42)

    Q_bf16 = torch.randn(q_len, num_heads, HD, dtype=torch.bfloat16)
    K_bf16 = torch.randn(kv_len, num_heads, HD, dtype=torch.bfloat16)
    V_bf16 = torch.randn(kv_len, num_heads, HD, dtype=torch.bfloat16)

    # ── bf16 correctness ─────────────────────────────────────────────────────
    ref_bf16 = cpu_sdp_ref(Q_bf16.float(), K_bf16.float(), V_bf16.float(), scale)

    Q_xpu = Q_bf16.unsqueeze(0).to(DEVICE)
    K_xpu = K_bf16.unsqueeze(0).to(DEVICE)
    V_xpu = V_bf16.unsqueeze(0).to(DEVICE)

    out_bf16 = sycl_kernels.sdp(Q_xpu, K_xpu, V_xpu)
    check("sdp(bf16)", out_bf16.squeeze(0).float().cpu(), ref_bf16, thresh=0.15)

    # ── fp16 correctness ─────────────────────────────────────────────────────
    Q_fp16 = Q_bf16.half()
    K_fp16 = K_bf16.half()
    V_fp16 = V_bf16.half()
    ref_fp16 = cpu_sdp_ref(Q_fp16.float(), K_fp16.float(), V_fp16.float(), scale)

    Q_xpu16 = Q_fp16.unsqueeze(0).to(DEVICE)
    K_xpu16 = K_fp16.unsqueeze(0).to(DEVICE)
    V_xpu16 = V_fp16.unsqueeze(0).to(DEVICE)

    out_fp16 = sycl_kernels.sdp(Q_xpu16, K_xpu16, V_xpu16)
    check("sdp(fp16)", out_fp16.squeeze(0).float().cpu(), ref_fp16, thresh=0.05)

    # ── bf16 perf ────────────────────────────────────────────────────────────
    bufs = [
        (
            Q_bf16.unsqueeze(0).to(DEVICE),
            K_bf16.unsqueeze(0).to(DEVICE),
            V_bf16.unsqueeze(0).to(DEVICE),
        )
        for _ in range(2)
    ]

    for _ in range(warmup):
        for q, k, v in bufs:
            sycl_kernels.sdp(q, k, v)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for i in range(iters):
        q, k, v = bufs[i % 2]
        sycl_kernels.sdp(q, k, v)
    torch.xpu.synchronize()
    elapsed_bf16 = (time.perf_counter() - t0) / iters * 1000

    flops = 4.0 * q_len * kv_len * num_heads * HD
    tflops_bf16 = flops / (elapsed_bf16 / 1000) / 1e12
    print(f"\n  sdp(bf16) perf:  {elapsed_bf16:.2f} ms/iter  |  {tflops_bf16:.1f} TFLOPS")

    # ── fp16 perf ────────────────────────────────────────────────────────────
    bufs16 = [
        (
            Q_fp16.unsqueeze(0).to(DEVICE),
            K_fp16.unsqueeze(0).to(DEVICE),
            V_fp16.unsqueeze(0).to(DEVICE),
        )
        for _ in range(2)
    ]

    for _ in range(warmup):
        for q, k, v in bufs16:
            sycl_kernels.sdp(q, k, v)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for i in range(iters):
        q, k, v = bufs16[i % 2]
        sycl_kernels.sdp(q, k, v)
    torch.xpu.synchronize()
    elapsed_fp16 = (time.perf_counter() - t0) / iters * 1000

    tflops_fp16 = flops / (elapsed_fp16 / 1000) / 1e12
    print(f"  sdp(fp16) perf:  {elapsed_fp16:.2f} ms/iter  |  {tflops_fp16:.1f} TFLOPS")

    # ── torch SDPA baseline ──────────────────────────────────────────────────
    Q_bhld = Q_xpu.permute(0, 2, 1, 3).contiguous()
    K_bhld = K_xpu.permute(0, 2, 1, 3).contiguous()
    V_bhld = V_xpu.permute(0, 2, 1, 3).contiguous()

    for _ in range(warmup):
        torch.nn.functional.scaled_dot_product_attention(Q_bhld, K_bhld, V_bhld)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        torch.nn.functional.scaled_dot_product_attention(Q_bhld, K_bhld, V_bhld)
    torch.xpu.synchronize()
    elapsed_sdpa = (time.perf_counter() - t0) / iters * 1000

    tflops_sdpa = flops / (elapsed_sdpa / 1000) / 1e12
    print(f"  torch SDPA bf16: {elapsed_sdpa:.2f} ms/iter  |  {tflops_sdpa:.1f} TFLOPS")
    print(f"\n  Speedup sdp(bf16) vs torch SDPA: {elapsed_sdpa / elapsed_bf16:.2f}x")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"XPU device: {torch.xpu.get_device_name(0)}")

    run_test("Self-attn  14040x14040", q_len=14040, kv_len=14040)
    run_test("Cross-attn 14040x512 ", q_len=14040, kv_len=512)

    print("\nAll tests done.")
