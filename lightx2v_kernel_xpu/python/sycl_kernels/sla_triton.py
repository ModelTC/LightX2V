"""Triton-XPU implementation detail for fused SLA block attention."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_block_attention_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    lut_ptr,
    out_ptr,
    q_len: tl.constexpr,
    kv_len: tl.constexpr,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    q_blocks: tl.constexpr,
    topk: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    qb = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // q_heads
    q_head = bh - batch * q_heads
    kv_head = q_head // (q_heads // kv_heads)

    offs_q = qb * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, HEAD_DIM)

    # Internally Q/K/V are BHLD so every per-head matrix tile is contiguous.
    # Direct BLHD access makes successive rows H*D apart and prevents Triton
    # XPU from forming efficient block loads / DPAS operands.
    q_base = (batch * q_heads + q_head) * q_len * HEAD_DIM
    kv_base = (batch * kv_heads + kv_head) * kv_len * HEAD_DIM
    q_ptrs = q_ptr + q_base + offs_q[:, None] * HEAD_DIM + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_q[:, None] < q_len, other=0.0)

    row_max = tl.full([BLOCK_Q], -float("inf"), tl.float32)
    row_sum = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], tl.float32)
    lut_base = (bh * q_blocks + qb) * topk
    log2e: tl.constexpr = 1.4426950408889634

    for slot in tl.range(0, topk, num_stages=2):
        key_block = tl.load(lut_ptr + lut_base + slot)
        key_pos = key_block * BLOCK_K + offs_k
        valid_k = key_pos < kv_len
        k_ptrs = k_ptr + kv_base + key_pos[:, None] * HEAD_DIM + offs_d[None, :]
        v_ptrs = v_ptr + kv_base + key_pos[:, None] * HEAD_DIM + offs_d[None, :]
        k_tile = tl.load(k_ptrs, mask=valid_k[:, None], other=0.0)
        v_tile = tl.load(v_ptrs, mask=valid_k[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k_tile)).to(tl.float32) * (scale * log2e)
        scores = tl.where(valid_k[None, :], scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        alpha = tl.exp2(row_max - new_max)
        probs = tl.exp2(scores - new_max[:, None])

        acc *= alpha[:, None]
        acc += tl.dot(probs.to(v_tile.dtype), v_tile)
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        row_max = new_max

    acc /= row_sum[:, None]
    out_ptrs = out_ptr + q_base + offs_q[:, None] * HEAD_DIM + offs_d[None, :]
    tl.store(out_ptrs, acc, mask=offs_q[:, None] < q_len)


def launch_sparse_block_attention(q, k, v, lut, block_q, block_k, scale):
    batch, q_len, q_heads, head_dim = q.shape
    kv_len, kv_heads = k.shape[1], k.shape[2]
    q_blocks, topk = lut.shape[2], lut.shape[3]
    q_bhld = q.permute(0, 2, 1, 3).contiguous()
    k_bhld = k.permute(0, 2, 1, 3).contiguous()
    v_bhld = v.permute(0, 2, 1, 3).contiguous()
    output = torch.empty_like(q_bhld)
    grid = (q_blocks, batch * q_heads)
    _sparse_block_attention_fwd[grid](
        q_bhld,
        k_bhld,
        v_bhld,
        lut,
        output,
        q_len,
        kv_len,
        q_heads,
        kv_heads,
        q_blocks,
        topk,
        scale,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        HEAD_DIM=head_dim,
        num_warps=8,
        num_stages=3,
    )
    return output.permute(0, 2, 1, 3)
