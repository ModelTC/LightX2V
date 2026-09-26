"""DreamZero's 5120-wide RMSNorm with the PyTorch 2.10 CUDA reduction order.

This is deliberately shape/version limited: matching a different reduction
schedule requires more than preserving the mathematical RMSNorm formula.
"""

import torch
import triton
import triton.language as tl


def can_use_torch_rms_norm_5120(x, weight, weight_diff):
    # Reduce.cuh changed its warp reduction order between PyTorch releases.
    if torch.__version__.split(".")[:2] != ["2", "10"] or torch.version.hip is not None:
        return False
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[1] == 5120
        and 16 <= x.shape[0]
        and x.numel() < 2**31
        and x.is_contiguous()
    ):
        return False
    if weight is None or not (
        weight.device == x.device
        and weight.dtype == x.dtype
        and weight.shape == (5120,)
        and weight.is_contiguous()
    ):
        return False
    return weight_diff is None or (
        weight_diff.device == x.device
        and weight_diff.dtype == x.dtype
        and weight_diff.shape in ((), (1,), (5120,))
        and weight_diff.is_contiguous()
    )


@triton.jit
def _torch_rms_norm_5120(
    X,
    W,
    WD,
    Y,
    ROWS: tl.constexpr,
    EPS: tl.constexpr,
    HAS_DIFF: tl.constexpr,
    SCALAR_DIFF: tl.constexpr,
):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    offsets = tl.arange(0, 128)
    accumulators = tl.full((4, 128), 0.0, tl.float32)
    # Reduce.cuh input_vectorized_thread_reduce_impl: 32 lanes, four
    # independent accumulators per lane, 40 iterations at stride 128.
    # Do not replace this loop with a generic reduction of all 5120 values.
    for chunk in range(40):
        x = tl.load(X + rows[:, None] * 5120 + chunk * 128 + offsets[None, :], rows[:, None] < ROWS, 0).to(tl.float32)
        accumulators = accumulators + x * x
    even, odd = tl.split(tl.reshape(accumulators, (4, 32, 2, 2)))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    # Sequential accumulator combine, then descending-offset warp reduction.
    total = tl.sum(((a0 + a1) + a2) + a3, axis=1)
    mean = total * (1.0 / 5120)
    rstd = tl.rsqrt(mean + EPS)

    columns = tl.arange(0, 8192)
    mask = (rows[:, None] < ROWS) & (columns[None, :] < 5120)
    x = tl.load(X + rows[:, None] * 5120 + columns[None, :], mask, 0).to(tl.float32)
    weight = tl.load(W + columns, columns < 5120, 0)
    if HAS_DIFF:
        if SCALAR_DIFF:
            delta = tl.load(WD)
        else:
            delta = tl.load(WD + columns, columns < 5120, 0)
        # Match RMSWeight._get_actual_weight's BF16 addition before multiply.
        weight = (weight.to(tl.float32) + delta.to(tl.float32)).to(weight.dtype)
    normalized = (x * rstd[:, None]).to(weight.dtype)
    y = normalized.to(tl.float32) * weight.to(tl.float32)
    tl.store(Y + rows[:, None] * 5120 + columns[None, :], y, mask)


def torch_rms_norm_5120(x, weight, weight_diff, eps):
    """One launch including weight delta; caller must check the support guard."""
    output = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _torch_rms_norm_5120[(triton.cdiv(x.shape[0], 4),)](
            x,
            weight,
            weight if weight_diff is None else weight_diff,
            output,
            x.shape[0],
            eps,
            weight_diff is not None,
            weight_diff is not None and weight_diff.numel() == 1,
            num_warps=4,
            # Both square/accumulate and mean/epsilon must round separately.
            enable_fp_fusion=False,
        )
    return output
