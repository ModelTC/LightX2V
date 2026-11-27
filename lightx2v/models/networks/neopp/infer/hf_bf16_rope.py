"""Fused generation Q/K normalization and rotary encoding with BF16 rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _round_bf16(v):
    return v.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _torch_mean_square(x, i, HALF: tl.constexpr):
    if HALF == 64:
        # ATen's contiguous 64-wide CUDA reduction first combines the two
        # values per lane, then shuffles by 1, 2, 4, 8, 16.
        acc = x * x
        acc += tl.gather(acc, i ^ 32, axis=0)
        for shift in tl.static_range(5):
            acc += tl.gather(acc, i ^ (1 << shift), axis=0)
        return acc / 64
    else:
        return tl.sum(x * x, axis=0) / HALF


@triton.jit
def _norm_rotate(x, partner, c, s, sign):
    return _round_bf16(x * c) + sign * _round_bf16(partner * s)


@triton.jit
def _apply_one(ptr, w_t, w_hw, ct, st, ch, sh, cw, sw, t, h, HEADS: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
    # Each program handles a complete token/head pair; HF normalizes time
    # and spatial components separately, with independent trainable weights.
    half: tl.constexpr = D // 2
    quarter: tl.constexpr = D // 4
    eighth: tl.constexpr = D // 8
    i = tl.arange(0, half)
    offset = (t * HEADS + h) * D
    a = tl.load(ptr + offset + i).to(tl.float32)
    b = tl.load(ptr + offset + half + i).to(tl.float32)
    na = _round_bf16(a * tl.rsqrt(_torch_mean_square(a, i, half) + EPS))
    nb = _round_bf16(b * tl.rsqrt(_torch_mean_square(b, i, half) + EPS))
    a = _round_bf16(na * tl.load(w_t + i).to(tl.float32))
    b = _round_bf16(nb * tl.load(w_hw + i).to(tl.float32))
    cr = tl.load(ct + t * half + i).to(tl.float32)
    sr = tl.load(st + t * half + i).to(tl.float32)
    time = _norm_rotate(a, tl.gather(a, i ^ quarter, axis=0), cr, sr, tl.where(i < quarter, -1.0, 1.0))
    # cos/sin tables repeat the first half, matching rotate_half semantics.
    ix = t * quarter + i % quarter
    cr = tl.where(i < quarter, tl.load(ch + ix), tl.load(cw + ix)).to(tl.float32)
    sr = tl.where(i < quarter, tl.load(sh + ix), tl.load(sw + ix)).to(tl.float32)
    hw = _norm_rotate(b, tl.gather(b, i ^ eighth, axis=0), cr, sr, tl.where(i % quarter < eighth, -1.0, 1.0))
    tl.store(ptr + offset + i, time)
    tl.store(ptr + offset + half + i, hw)


@triton.jit
def _kernel(q, k, wqt, wqhw, wkt, wkhw, ct, st, ch, sh, cw, sw, QH: tl.constexpr, KH: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    if h < QH:
        _apply_one(q, wqt, wqhw, ct, st, ch, sh, cw, sw, t, h, QH, D, EPS)
    else:
        _apply_one(k, wkt, wkhw, ct, st, ch, sh, cw, sw, t, h - QH, KH, D, EPS)


def fused_hf_bf16_rope(q, k, wqt, wqhw, wkt, wkhw, ct, st, ch, sh, cw, sw, eps=1e-6):
    n, qh, d = q.shape
    kh = k.shape[1]
    assert q.is_contiguous() and k.is_contiguous() and q.dtype == k.dtype == torch.bfloat16
    assert d % 8 == 0 and d & (d - 1) == 0
    _kernel[(n, qh + kh)](
        q,
        k,
        wqt,
        wqhw,
        wkt,
        wkhw,
        ct.contiguous(),
        st.contiguous(),
        ch.contiguous(),
        sh.contiguous(),
        cw.contiguous(),
        sw.contiguous(),
        qh,
        kh,
        d,
        eps,
        num_warps=1,
        enable_fp_fusion=False,
    )
