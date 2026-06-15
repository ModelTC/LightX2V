import argparse

import torch

from lightx2v_kernel.gemm import (
    cutlass_scaled_mxfp8_mm,
    cutlass_scaled_mxfp8_mm_residual_gate,
    scaled_mxfp8_quant,
)


def first_visible_sm120_device():
    if not torch.cuda.is_available():
        return None
    for device_index in range(torch.cuda.device_count()):
        major, _minor = torch.cuda.get_device_capability(device_index)
        if major == 12:
            return torch.device("cuda", device_index)
    return None


def bench_cuda(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def make_inputs(m, k, n):
    activation = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.5
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.5
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.1
    activation_quant, activation_scale = scaled_mxfp8_quant(activation)
    weight_quant, weight_scale = scaled_mxfp8_quant(weight)
    alpha = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    residual = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    gate_1d = torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.25
    gate_2d = gate_1d.expand(m, n).contiguous()
    return activation_quant, weight_quant, activation_scale, weight_scale, alpha, bias, residual, gate_1d, gate_2d


def assert_close_enough(actual, expected):
    actual_f = actual.float().flatten()
    expected_f = expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(actual_f, expected_f, dim=0).item()
    max_abs = (actual_f - expected_f).abs().max().item()
    if cosine <= 0.999 or max_abs >= 0.08:
        raise AssertionError(f"cosine={cosine}, max_abs={max_abs}")


def run_shape(m, k, n, warmup, iters):
    activation_quant, weight_quant, activation_scale, weight_scale, alpha, bias, residual, gate_1d, gate_2d = make_inputs(m, k, n)

    gemm_out = cutlass_scaled_mxfp8_mm(
        activation_quant,
        weight_quant,
        activation_scale,
        weight_scale,
        alpha,
        bias=bias,
    )
    baseline = residual + gemm_out * gate_2d
    fallback = cutlass_scaled_mxfp8_mm_residual_gate(
        activation_quant,
        weight_quant,
        activation_scale,
        weight_scale,
        alpha,
        residual=residual.clone(),
        gate=gate_2d,
        bias=bias,
    )
    torch.cuda.synchronize()
    assert_close_enough(fallback, baseline)

    gemm_ms = bench_cuda(
        lambda: cutlass_scaled_mxfp8_mm(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            bias=bias,
        ),
        warmup,
        iters,
    )

    gemm_out = cutlass_scaled_mxfp8_mm(
        activation_quant,
        weight_quant,
        activation_scale,
        weight_scale,
        alpha,
        bias=bias,
    )
    torch_pointwise_ms = bench_cuda(lambda: residual + gemm_out * gate_2d, warmup, iters)

    fallback_2d_ms = bench_cuda(
        lambda: cutlass_scaled_mxfp8_mm_residual_gate(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            residual=residual.clone(),
            gate=gate_2d,
            bias=bias,
        ),
        warmup,
        iters,
    )

    fast_1d_ms = bench_cuda(
        lambda: cutlass_scaled_mxfp8_mm_residual_gate(
            activation_quant,
            weight_quant,
            activation_scale,
            weight_scale,
            alpha,
            residual=residual.clone(),
            gate=gate_1d,
            bias=bias,
        ),
        warmup,
        iters,
    )

    baseline_ms = gemm_ms + torch_pointwise_ms
    pointwise_overhead = max(fallback_2d_ms - gemm_ms, 0.0)
    pointwise_saving = (torch_pointwise_ms - pointwise_overhead) / torch_pointwise_ms if torch_pointwise_ms else float("nan")
    return {
        "shape": f"{m}x{k}x{n}",
        "gemm_ms": gemm_ms,
        "torch_pointwise_ms": torch_pointwise_ms,
        "baseline_ms": baseline_ms,
        "fallback_2d_ms": fallback_2d_ms,
        "speedup": baseline_ms / fallback_2d_ms,
        "pointwise_saving": pointwise_saving,
        "fast_1d_ms": fast_1d_ms,
        "gap_vs_1d": fallback_2d_ms / fast_1d_ms,
    }


def parse_shape(value):
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be MxKxN")
    return tuple(int(part) for part in parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=parse_shape, default=[])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    device = first_visible_sm120_device()
    if device is None:
        raise RuntimeError("MXFP8 fused FFN benchmark requires a visible SM120/SM120a CUDA device")
    torch.cuda.set_device(device)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    shapes = args.shape or [
        (257, 256, 384),
        (1024, 256, 384),
        (4096, 256, 384),
    ]
    rows = [run_shape(m, k, n, args.warmup, args.iters) for m, k, n in shapes]

    print(
        "shape          gemm_ms  torch_pw_ms  baseline_ms  2d_fallback_ms  "
        "speedup  pw_saving  1d_fast_ms  gap_vs_1d"
    )
    for row in rows:
        print(
            f"{row['shape']:<14}"
            f"{row['gemm_ms']:>8.4f}"
            f"{row['torch_pointwise_ms']:>13.4f}"
            f"{row['baseline_ms']:>13.4f}"
            f"{row['fallback_2d_ms']:>16.4f}"
            f"{row['speedup']:>9.3f}"
            f"{row['pointwise_saving']:>11.3f}"
            f"{row['fast_1d_ms']:>12.4f}"
            f"{row['gap_vs_1d']:>11.3f}"
        )


if __name__ == "__main__":
    main()
