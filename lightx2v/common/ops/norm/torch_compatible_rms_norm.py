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
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 2 and x.shape[1] == 5120 and 16 <= x.shape[0] and x.numel() < 2**31 and x.is_contiguous()):
        return False
    if weight is None or not (weight.device == x.device and weight.dtype == x.dtype and weight.shape == (5120,) and weight.is_contiguous()):
        return False
    return weight_diff is None or (weight_diff.device == x.device and weight_diff.dtype == x.dtype and weight_diff.shape in ((), (1,), (5120,)) and weight_diff.is_contiguous())


@triton.jit
def _rope_pair(ye, yo, c, s):
    # flashinfer vec_apply_llama_rope_cos_sin_interleave_reuse_half (fp32):
    #   out[2p]   = x[2p]   * cos[p] + (-x[2p+1]) * sin[p]
    #   out[2p+1] = x[2p+1] * cos[p] +   x[2p]    * sin[p]
    # nvcc contracts the first product into an FMA (verified bitwise).
    oe = tl.fma(ye, c, (-yo) * s)
    oo = tl.fma(yo, c, ye * s)
    return oe, oo


@triton.jit
def _torch_rms_norm_5120(
    X,
    W,
    WD,
    Y,
    COS_SIN,
    ROWS: tl.constexpr,
    EPS: tl.constexpr,
    HAS_DIFF: tl.constexpr,
    SCALAR_DIFF: tl.constexpr,
    ROPE: tl.constexpr,
):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    offsets = tl.arange(0, 128)
    row_mask = rows[:, None] < ROWS
    accumulators = tl.full((4, 128), 0.0, tl.float32)
    # Reduce.cuh input_vectorized_thread_reduce_impl: 32 lanes, four
    # independent accumulators per lane, 40 iterations at stride 128.
    # Do not replace this loop with a generic reduction of all 5120 values.
    for chunk in range(40):
        x = tl.load(X + rows[:, None] * 5120 + chunk * 128 + offsets[None, :], row_mask, 0).to(tl.float32)
        accumulators = accumulators + x * x
    even, odd = tl.split(tl.reshape(accumulators, (4, 32, 2, 2)))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    # Sequential accumulator combine, then descending-offset warp reduction.
    total = tl.sum(((a0 + a1) + a2) + a3, axis=1)
    mean = total * (1.0 / 5120)
    rstd = tl.rsqrt(mean + EPS)

    # Elementwise pass, one 128-wide head at a time (the rows are L2 resident).
    if SCALAR_DIFF:
        delta = tl.load(WD)
    for head in range(40):
        columns = head * 128 + offsets
        x = tl.load(X + rows[:, None] * 5120 + columns[None, :], row_mask, 0).to(tl.float32)
        weight = tl.load(W + columns)
        if HAS_DIFF:
            if not SCALAR_DIFF:
                delta = tl.load(WD + columns)
            # Match RMSWeight._get_actual_weight's BF16 addition before multiply.
            weight = (weight.to(tl.float32) + delta.to(tl.float32)).to(weight.dtype)
        normalized = (x * rstd[:, None]).to(weight.dtype)
        y = normalized.to(tl.float32) * weight.to(tl.float32)
        if ROPE:
            # Interleaved RoPE on the BF16-rounded RMSNorm output; position = row.
            ye, yo = tl.split(tl.reshape(y.to(weight.dtype).to(tl.float32), (4, 64, 2)))
            pairs = tl.arange(0, 64)
            cs_row = COS_SIN + rows[:, None].to(tl.int64) * 128 + pairs[None, :]
            c = tl.load(cs_row, row_mask, 0.0)
            s = tl.load(cs_row + 64, row_mask, 0.0)
            oe, oo = _rope_pair(ye, yo, c, s)
            y = tl.reshape(tl.join(oe, oo), (4, 128))
        tl.store(Y + rows[:, None] * 5120 + columns[None, :], y, row_mask)


def torch_rms_norm_5120(x, weight, weight_diff, eps, out=None, rope_cos_sin=None):
    """One launch including weight delta; caller must check the support guard.

    ``out`` (optional) must be a contiguous tensor shaped and typed like ``x``.
    ``rope_cos_sin`` (optional, fp32 [rows, 128] = [cos(64) | sin(64)] for
    position = row) additionally applies interleaved RoPE over 40 heads of
    128 like flashinfer's apply_rope_with_cos_sin_cache_inplace.
    """
    if out is None:
        output = torch.empty_like(x)
    else:
        if not (out.shape == x.shape and out.dtype == x.dtype and out.device == x.device and out.is_contiguous()):
            raise ValueError("torch_rms_norm_5120 out must be contiguous and match x")
        output = out
    rope = rope_cos_sin is not None
    if rope and not (rope_cos_sin.dtype == torch.float32 and rope_cos_sin.is_contiguous() and rope_cos_sin.shape == (x.shape[0], 128) and rope_cos_sin.device == x.device):
        raise ValueError("rope_cos_sin must be a contiguous fp32 [rows, 128] tensor")
    with torch.cuda.device(x.device):
        _torch_rms_norm_5120[(triton.cdiv(x.shape[0], 4),)](
            x,
            weight,
            weight if weight_diff is None else weight_diff,
            output,
            rope_cos_sin if rope else x,
            x.shape[0],
            eps,
            weight_diff is not None,
            weight_diff is not None and weight_diff.numel() == 1,
            rope,
            num_warps=4,
            # Both square/accumulate and mean/epsilon must round separately.
            enable_fp_fusion=False,
        )
    return output
