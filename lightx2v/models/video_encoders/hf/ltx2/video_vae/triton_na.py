# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton 3D neighborhood attention with NATTEN ``na3d`` semantics.

Vendored from comfy-kitchen via the Apache-2.0 LTX-2 fallback implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_NEG_INF = tl.constexpr(-3.0e38)


@triton.jit
def _na3d_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    t_size,
    h_size,
    w_size,
    num_heads,
    s_b,
    s_t,
    s_h,
    s_w,
    s_n,
    scale,
    kt: tl.constexpr,
    kh: tl.constexpr,
    kw: tl.constexpr,
    causal_t: tl.constexpr,
    causal_h: tl.constexpr,
    causal_w: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_th = tl.program_id(1)
    pid_bn = tl.program_id(2)
    t_q = pid_th // h_size
    h_q = pid_th % h_size
    base = (pid_bn // num_heads) * s_b + (pid_bn % num_heads) * s_n

    w_off = pid_w * block_q + tl.arange(0, block_q)
    w_valid = w_off < w_size
    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd
    q_ptrs = q_ptr + base + t_q * s_t + h_q * s_h + w_off[:, None] * s_w + d_off[None]
    q_block = tl.load(q_ptrs, mask=w_valid[:, None] & d_mask[None], other=0.0)

    if causal_t:
        t_lo = tl.maximum(t_q - kt + 1, 0)
        t_hi = t_q + 1
    else:
        t_lo = tl.minimum(tl.maximum(t_q - kt // 2, 0), t_size - kt)
        t_hi = t_lo + kt
    if causal_h:
        h_lo = tl.maximum(h_q - kh + 1, 0)
        h_hi = h_q + 1
    else:
        h_lo = tl.minimum(tl.maximum(h_q - kh // 2, 0), h_size - kh)
        h_hi = h_lo + kh

    w_q = tl.where(w_valid, w_off, w_size - 1)
    if causal_w:
        w_start = tl.maximum(w_q - kw + 1, 0)
        w_end = w_q + 1
        first = tl.minimum(pid_w * block_q, w_size - 1)
        last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
        w_lo = tl.maximum(first - kw + 1, 0)
        w_hi = last + 1
    else:
        w_start = tl.minimum(tl.maximum(w_q - kw // 2, 0), w_size - kw)
        w_end = w_start + kw
        first = tl.minimum(pid_w * block_q, w_size - 1)
        last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
        w_lo = tl.minimum(tl.maximum(first - kw // 2, 0), w_size - kw)
        w_hi = tl.minimum(tl.maximum(last - kw // 2, 0), w_size - kw) + kw

    row_max = tl.full((block_q,), _NEG_INF, tl.float32)
    row_sum = tl.zeros((block_q,), tl.float32)
    accumulator = tl.zeros((block_q, hd_pad), tl.float32)
    for t_key in range(t_lo, t_hi):
        for h_key in range(h_lo, h_hi):
            plane = base + t_key * s_t + h_key * s_h
            for w_key_start in range(w_lo, w_hi, block_k):
                w_key = w_key_start + tl.arange(0, block_k)
                key_mask = w_key < w_hi
                offsets = plane + w_key[:, None] * s_w + d_off[None]
                value_mask = key_mask[:, None] & d_mask[None]
                key_block = tl.load(k_ptr + offsets, mask=value_mask, other=0.0)
                if is_fp32:
                    score = tl.dot(q_block, tl.trans(key_block), input_precision="ieee") * scale
                else:
                    score = tl.dot(q_block, tl.trans(key_block)) * scale
                visible = (w_key[None] >= w_start[:, None]) & (w_key[None] < w_end[:, None]) & key_mask[None]
                score = tl.where(visible, score, _NEG_INF)
                new_max = tl.maximum(row_max, tl.max(score, 1))
                alpha = tl.exp(row_max - new_max)
                probability = tl.exp(score - new_max[:, None])
                row_sum = row_sum * alpha + tl.sum(probability, 1)
                value_block = tl.load(v_ptr + offsets, mask=value_mask, other=0.0)
                if is_fp32:
                    accumulator = accumulator * alpha[:, None] + tl.dot(probability, value_block, input_precision="ieee")
                else:
                    accumulator = accumulator * alpha[:, None] + tl.dot(probability.to(value_block.dtype), value_block)
                row_max = new_max

    output = accumulator / tl.maximum(row_sum, 1e-30)[:, None]
    output_ptrs = out_ptr + base + t_q * s_t + h_q * s_h + w_off[:, None] * s_w + d_off[None]
    tl.store(output_ptrs, output.to(out_ptr.dtype.element_ty), mask=w_valid[:, None] & d_mask[None])


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
    scale: float = 1.0,
) -> torch.Tensor:
    batch, time, height, width, num_heads, head_dim = q.shape
    kt, kh, kw = (min(kernel, dim) for kernel, dim in zip(kernel_size, (time, height, width)))
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    output = torch.empty_like(q)
    head_dim_padded = max(16, triton.next_power_of_2(head_dim))
    block_q = 16
    block_k = max(16, min(32, triton.next_power_of_2(min(width, block_q + kw))))
    grid = (triton.cdiv(width, block_q), time * height, batch * num_heads)
    _na3d_kernel[grid](
        q,
        k,
        v,
        output,
        time,
        height,
        width,
        num_heads,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q.stride(4),
        scale,
        kt=kt,
        kh=kh,
        kw=kw,
        causal_t=False,
        causal_h=False,
        causal_w=False,
        hd=head_dim,
        hd_pad=head_dim_padded,
        block_q=block_q,
        block_k=block_k,
        is_fp32=q.dtype == torch.float32,
        num_warps=4,
    )
    return output


__all__ = ["na3d"]
