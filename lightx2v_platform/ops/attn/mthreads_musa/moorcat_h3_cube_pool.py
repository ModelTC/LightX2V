# SPDX-License-Identifier: Apache-2.0
"""Fused cube-order Q/K pooling for MiniMax-H3 block-sparse attention."""

import torch
import triton
import triton.language as tl


_HEAD_DIM = 128
_MAX_LABEL_ROWS = 128


@triton.jit
def _cube_pool_kernel(
    q,
    k,
    sorted_indices,
    label_offsets,
    q_pool,
    k_pool,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    pool_label_stride,
    pool_head_stride,
    MAX_LABEL_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    label = tl.program_id(0)
    head = tl.program_id(1)
    dim_block = tl.program_id(2)
    rows = tl.arange(0, MAX_LABEL_ROWS)
    dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    begin = tl.load(label_offsets + label)
    end = tl.load(label_offsets + label + 1)
    row_mask = begin + rows < end
    token = tl.load(sorted_indices + begin + rows, mask=row_mask, other=0)
    mask = row_mask[:, None] & (dims[None, :] < HEAD_DIM)
    q_values = tl.load(
        q + token[:, None] * q_token_stride + head * q_head_stride + dims[None, :],
        mask=mask,
        other=0.0,
    )
    k_values = tl.load(
        k + token[:, None] * k_token_stride + head * k_head_stride + dims[None, :],
        mask=mask,
        other=0.0,
    )
    scale = 1.0 / (end - begin)
    output_offset = label * pool_label_stride + head * pool_head_stride + dims
    output_mask = dims < HEAD_DIM
    tl.store(q_pool + output_offset, tl.sum(q_values, axis=0) * scale, mask=output_mask)
    tl.store(k_pool + output_offset, tl.sum(k_values, axis=0) * scale, mask=output_mask)


def cube_pool_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    sorted_indices: torch.Tensor,
    label_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = label_offsets.numel() - 1
    q_pool = torch.empty(labels, q.shape[1], q.shape[2], dtype=q.dtype, device=q.device)
    k_pool = torch.empty_like(q_pool)
    _cube_pool_kernel[(labels, q.shape[1], triton.cdiv(_HEAD_DIM, 32))](
        q,
        k,
        sorted_indices,
        label_offsets,
        q_pool,
        k_pool,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        q_pool.stride(0),
        q_pool.stride(1),
        MAX_LABEL_ROWS=_MAX_LABEL_ROWS,
        HEAD_DIM=_HEAD_DIM,
        BLOCK_D=32,
        num_warps=4,
    )
    return q_pool, k_pool


__all__ = ["cube_pool_qk"]
