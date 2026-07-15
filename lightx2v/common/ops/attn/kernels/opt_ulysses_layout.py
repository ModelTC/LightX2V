"""Fused layout kernels for the ``ulysses-opt`` attention path.

The names intentionally mirror the two Ulysses communications:
- ``qkv_pre`` / ``qkv_post`` wrap the all-to-all before attention.
- ``attn_pre`` / ``attn_post`` wrap the all-to-all after attention.
- ``*_fp8`` variants keep the same layout contract, but return separate
  FP8 payload and FP32 scale tensors. Keeping these as separate collectives
  matches the faster NCCL message shape used by the legacy implementation.

The public pre/post wrappers each launch one Triton kernel on the hot path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv_pre_kernel(
    q,
    k,
    v,
    out,
    total_rows,
    q_img_start,
    kv_img_start,
    img_len: tl.constexpr,
    world_size: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    full_heads: tl.constexpr,
    head_offset: tl.constexpr,
    head_stride: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_d)
    row_mask = rows < total_rows
    col_mask = cols < hidden_dims
    mask = row_mask[:, None] & col_mask[None, :]

    # Tile layout: rows index [dst_rank, local_img_s, shard_head], cols index hidden.
    # The q/k/v dimension is handled inside this one program so the pre pass still
    # emits the exact rank-major [W, S_img, 3, H/W, D] all-to-all layout.
    rows_per_rank: tl.constexpr = img_len * shard_heads
    rank = rows // rows_per_rank
    row_in_rank = rows - rank * rows_per_rank
    h = row_in_rank % shard_heads
    s = row_in_rank // shard_heads

    src_head = head_offset + rank * head_stride + h
    q_src_s = q_img_start + s
    kv_src_s = kv_img_start + s
    q_offsets = (q_src_s[:, None] * full_heads + src_head[:, None]) * hidden_dims + cols[None, :]
    kv_offsets = (kv_src_s[:, None] * full_heads + src_head[:, None]) * hidden_dims + cols[None, :]

    q_row = s * 3 * shard_heads + h
    out_base = rank[:, None] * (img_len * 3 * shard_heads * hidden_dims) + q_row[:, None] * hidden_dims + cols[None, :]
    tl.store(out + out_base, tl.load(q + q_offsets, mask=mask), mask=mask)
    tl.store(out + out_base + shard_heads * hidden_dims, tl.load(k + kv_offsets, mask=mask), mask=mask)
    tl.store(out + out_base + 2 * shard_heads * hidden_dims, tl.load(v + kv_offsets, mask=mask), mask=mask)


@triton.jit
def _qkv_pre_heads_kernel(
    q,
    k,
    v,
    out,
    total_rows,
    q_img_start,
    kv_img_start,
    img_len: tl.constexpr,
    world_size: tl.constexpr,
    head_count: tl.constexpr,
    hidden_dims: tl.constexpr,
    full_heads: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_d)
    row_mask = rows < total_rows
    col_mask = cols < hidden_dims
    mask = row_mask[:, None] & col_mask[None, :]

    # Head-parallel pre keeps communication buffers independent per local head:
    # out[local_head] is a contiguous [W, S_img, 3, 1, D] split buffer.
    # One launch fills all local_head slabs, but each slab is still sent by its
    # own all_to_all_single in the caller.
    rows_per_head: tl.constexpr = world_size * img_len
    local_head = rows // rows_per_head
    row_in_head = rows - local_head * rows_per_head
    rank = row_in_head // img_len
    s = row_in_head - rank * img_len

    src_head = local_head + rank * head_count
    q_src_s = q_img_start + s
    kv_src_s = kv_img_start + s
    q_offsets = (q_src_s[:, None] * full_heads + src_head[:, None]) * hidden_dims + cols[None, :]
    kv_offsets = (kv_src_s[:, None] * full_heads + src_head[:, None]) * hidden_dims + cols[None, :]

    head_stride_elems: tl.constexpr = world_size * img_len * 3 * hidden_dims
    rank_stride_elems: tl.constexpr = img_len * 3 * hidden_dims
    q_base = local_head[:, None] * head_stride_elems + rank[:, None] * rank_stride_elems + s[:, None] * 3 * hidden_dims + cols[None, :]
    tl.store(out + q_base, tl.load(q + q_offsets, mask=mask), mask=mask)
    tl.store(out + q_base + hidden_dims, tl.load(k + kv_offsets, mask=mask), mask=mask)
    tl.store(out + q_base + 2 * hidden_dims, tl.load(v + kv_offsets, mask=mask), mask=mask)


@triton.jit
def _qkv_post_kernel(
    qkv,
    q_full,
    k_full,
    v_full,
    out_q,
    out_k,
    out_v,
    total_rows,
    txt_start,
    txt_len,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    full_heads: tl.constexpr,
    head_offset: tl.constexpr,
    head_stride: tl.constexpr,
    img_first: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_d)
    row_mask = rows < total_rows
    col_mask = cols < hidden_dims
    mask = row_mask[:, None] & col_mask[None, :]

    # Tile layout: rows index [global_or_text_s, local_shard_head], cols index hidden.
    # This mirrors the FP8 post kernel without dequantization: one program now handles
    # several complete hidden rows, so each row's image/text routing is computed once.
    h = rows % shard_heads
    out_s = rows // shard_heads

    final_len = total_rows // shard_heads
    global_img_len = final_len - txt_len
    local_img_len = global_img_len // world_size
    if img_first:
        is_img = out_s < global_img_len
        img_global_s = out_s
        txt_s = out_s - global_img_len
    else:
        is_img = out_s >= txt_len
        img_global_s = out_s - txt_len
        txt_s = out_s

    img_rank = img_global_s // local_img_len
    img_s = img_global_s - img_rank * local_img_len
    qkv_base = ((img_rank[:, None] * local_img_len + img_s[:, None]) * 3 * shard_heads + h[:, None]) * hidden_dims + cols[None, :]

    txt_head = head_offset + cur_rank * head_stride + h
    txt_src_s = txt_start + txt_s
    txt_base = (txt_src_s[:, None] * full_heads + txt_head[:, None]) * hidden_dims + cols[None, :]

    img_mask = mask & is_img[:, None]
    q_img = tl.load(qkv + qkv_base, mask=img_mask, other=0.0)
    k_img = tl.load(qkv + qkv_base + shard_heads * hidden_dims, mask=img_mask, other=0.0)
    v_img = tl.load(qkv + qkv_base + 2 * shard_heads * hidden_dims, mask=img_mask, other=0.0)

    txt_mask = mask & ~is_img[:, None]
    q_txt = tl.load(q_full + txt_base, mask=txt_mask, other=0.0)
    k_txt = tl.load(k_full + txt_base, mask=txt_mask, other=0.0)
    v_txt = tl.load(v_full + txt_base, mask=txt_mask, other=0.0)

    out_offsets = rows[:, None] * hidden_dims + cols[None, :]
    tl.store(out_q + out_offsets, q_img + q_txt, mask=mask)
    tl.store(out_k + out_offsets, k_img + k_txt, mask=mask)
    tl.store(out_v + out_offsets, v_img + v_txt, mask=mask)


@triton.jit
def _qonly_qkv_post_kernel(
    qkv,
    k_full,
    v_full,
    out_q,
    out_k,
    out_v,
    total_rows,
    q_rows,
    txt_start,
    txt_len,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    full_heads: tl.constexpr,
    head_offset: tl.constexpr,
    head_stride: tl.constexpr,
    img_first: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_d)
    row_mask = rows < total_rows
    col_mask = cols < hidden_dims
    mask = row_mask[:, None] & col_mask[None, :]

    h = rows % shard_heads
    out_s = rows // shard_heads

    global_img_len = q_rows // shard_heads
    local_img_len = global_img_len // world_size
    kv_final_len = global_img_len + txt_len
    kv_rows = kv_final_len * shard_heads

    is_q_row = rows < q_rows
    q_img_rank = out_s // local_img_len
    q_img_s = out_s - q_img_rank * local_img_len
    q_row = q_img_s * 3 * shard_heads + h
    q_base = ((q_img_rank[:, None] * local_img_len + q_img_s[:, None]) * 3 * shard_heads + h[:, None]) * hidden_dims + cols[None, :]
    q_val = tl.load(qkv + q_base, mask=mask & is_q_row[:, None], other=0.0)
    tl.store(out_q + rows[:, None] * hidden_dims + cols[None, :], q_val, mask=mask & is_q_row[:, None])

    is_kv_row = rows < kv_rows
    if img_first:
        is_img = out_s < global_img_len
        img_global_s = out_s
        txt_s = out_s - global_img_len
    else:
        is_img = out_s >= txt_len
        img_global_s = out_s - txt_len
        txt_s = out_s

    img_rank = img_global_s // local_img_len
    img_s = img_global_s - img_rank * local_img_len
    img_row = img_s * 3 * shard_heads + h
    k_img_base = ((img_rank[:, None] * local_img_len + img_s[:, None]) * 3 * shard_heads + h[:, None] + shard_heads) * hidden_dims + cols[None, :]
    v_img_base = k_img_base + shard_heads * hidden_dims
    img_mask = mask & is_kv_row[:, None] & is_img[:, None]
    k_img = tl.load(qkv + k_img_base, mask=img_mask, other=0.0)
    v_img = tl.load(qkv + v_img_base, mask=img_mask, other=0.0)

    txt_head = head_offset + cur_rank * head_stride + h
    txt_src_s = txt_start + txt_s
    txt_base = (txt_src_s[:, None] * full_heads + txt_head[:, None]) * hidden_dims + cols[None, :]
    txt_mask = mask & is_kv_row[:, None] & ~is_img[:, None]
    k_txt = tl.load(k_full + txt_base, mask=txt_mask, other=0.0)
    v_txt = tl.load(v_full + txt_base, mask=txt_mask, other=0.0)

    kv_offsets = rows[:, None] * hidden_dims + cols[None, :]
    kv_mask = mask & is_kv_row[:, None]
    tl.store(out_k + kv_offsets, k_img + k_txt, mask=kv_mask)
    tl.store(out_v + kv_offsets, v_img + v_txt, mask=kv_mask)


@triton.jit
def _attn_pre_kernel(
    attn,
    out,
    total_rows,
    img_start,
    img_len: tl.constexpr,
    world_size: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_d)
    row_mask = rows < total_rows
    col_mask = cols < hidden_dims
    mask = row_mask[:, None] & col_mask[None, :]

    # Tile layout: rows index [dst_rank, shard_head, local_img_s], cols index hidden.
    # This keeps each output row contiguous while avoiding per-element rank/head math.
    s = rows % img_len
    tmp = rows // img_len
    h = tmp % shard_heads
    rank = tmp // shard_heads

    src_s = img_start + rank * img_len + s
    src_offsets = src_s[:, None] * (shard_heads * hidden_dims) + h[:, None] * hidden_dims + cols[None, :]
    out_offsets = rows[:, None] * hidden_dims + cols[None, :]
    tl.store(out + out_offsets, tl.load(attn + src_offsets, mask=mask), mask=mask)


@triton.jit
def _attn_post_kernel(
    attn,
    txt_attn,
    out,
    total_rows,
    txt_len,
    world_size: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    img_first: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_d)
    row_mask = rows < total_rows
    col_mask = cols < hidden_dims
    mask = row_mask[:, None] & col_mask[None, :]

    # Tile layout: rows index [local_img_or_text_s, rank, shard_head], cols index hidden.
    # Image rows come from the communicated buffer; text rows are copied from the
    # already all-gathered full-head text attention output in the same post pass.
    hidden_rows_per_seq = world_size * shard_heads
    h = rows % shard_heads
    tmp = rows // shard_heads
    rank = tmp % world_size
    out_s = tmp // world_size

    final_len = total_rows // hidden_rows_per_seq
    img_len = final_len - txt_len
    if img_first:
        is_img = out_s < img_len
        img_s = out_s
        txt_s = out_s - img_len
    else:
        is_img = out_s >= txt_len
        img_s = out_s - txt_len
        txt_s = out_s

    img_base = ((rank[:, None] * shard_heads + h[:, None]) * img_len + img_s[:, None]) * hidden_dims + cols[None, :]
    all_hidden = world_size * shard_heads * hidden_dims
    txt_base = txt_s[:, None] * all_hidden + (rank[:, None] * shard_heads + h[:, None]) * hidden_dims + cols[None, :]

    img_val = tl.load(attn + img_base, mask=mask & is_img[:, None], other=0.0)
    txt_val = tl.load(txt_attn + txt_base, mask=mask & ~is_img[:, None], other=0.0)
    out_offsets = rows[:, None] * hidden_dims + cols[None, :]
    tl.store(out + out_offsets, img_val + txt_val, mask=mask)


@triton.jit
def _qkv_pre_fp8_kernel(
    q,
    k,
    v,
    payload_fp8,
    scale_ptr,
    q_img_start,
    kv_img_start,
    img_len: tl.constexpr,
    world_size: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    full_heads: tl.constexpr,
    head_offset: tl.constexpr,
    head_stride: tl.constexpr,
    payload_elems_per_rank: tl.constexpr,
    scale_base_offset: tl.constexpr,
    scale_elems_per_rank: tl.constexpr,
    block_d: tl.constexpr,
):
    # One program handles the q/k/v triplet for one [dst_rank, local_img_s, shard_head].
    # That keeps the output layout unchanged while cutting launch work from 3 rows to 1.
    row = tl.program_id(0)
    offs = tl.arange(0, block_d)
    mask = offs < hidden_dims

    rows_per_rank: tl.constexpr = img_len * shard_heads
    rank = row // rows_per_rank
    row_in_rank = row - rank * rows_per_rank
    h = row_in_rank % shard_heads
    s = row_in_rank // shard_heads

    src_head = head_offset + rank * head_stride + h
    q_src_s = q_img_start + s
    kv_src_s = kv_img_start + s
    q_offsets = (q_src_s * full_heads + src_head) * hidden_dims + offs
    kv_offsets = (kv_src_s * full_heads + src_head) * hidden_dims + offs

    q_vals = tl.load(q + q_offsets, mask=mask, other=0.0).to(tl.float32)
    k_vals = tl.load(k + kv_offsets, mask=mask, other=0.0).to(tl.float32)
    v_vals = tl.load(v + kv_offsets, mask=mask, other=0.0).to(tl.float32)

    q_amax = tl.maximum(tl.max(tl.abs(q_vals), axis=0), 0.001953125)
    k_amax = tl.maximum(tl.max(tl.abs(k_vals), axis=0), 0.001953125)
    v_amax = tl.maximum(tl.max(tl.abs(v_vals), axis=0), 0.001953125)
    q_scale = q_amax / 448.0
    k_scale = k_amax / 448.0
    v_scale = v_amax / 448.0

    q_row = s * 3 * shard_heads + h
    k_row = q_row + shard_heads
    v_row = q_row + 2 * shard_heads
    q_payload = rank * payload_elems_per_rank + q_row * hidden_dims + offs
    k_payload = rank * payload_elems_per_rank + k_row * hidden_dims + offs
    v_payload = rank * payload_elems_per_rank + v_row * hidden_dims + offs
    tl.store(payload_fp8 + q_payload, (q_vals / q_scale).to(tl.float8e4nv), mask=mask)
    tl.store(payload_fp8 + k_payload, (k_vals / k_scale).to(tl.float8e4nv), mask=mask)
    tl.store(payload_fp8 + v_payload, (v_vals / v_scale).to(tl.float8e4nv), mask=mask)

    q_scale_offset = rank * scale_elems_per_rank + scale_base_offset + q_row
    k_scale_offset = rank * scale_elems_per_rank + scale_base_offset + k_row
    v_scale_offset = rank * scale_elems_per_rank + scale_base_offset + v_row
    tl.store(scale_ptr + q_scale_offset, q_scale)
    tl.store(scale_ptr + k_scale_offset, k_scale)
    tl.store(scale_ptr + v_scale_offset, v_scale)


@triton.jit
def _qkv_pre_heads_fp8_kernel(
    q,
    k,
    v,
    payload_fp8,
    scale_ptr,
    q_img_start,
    kv_img_start,
    img_len: tl.constexpr,
    world_size: tl.constexpr,
    head_count: tl.constexpr,
    hidden_dims: tl.constexpr,
    full_heads: tl.constexpr,
    payload_elems_per_rank: tl.constexpr,
    scale_base_offset: tl.constexpr,
    scale_elems_per_rank: tl.constexpr,
    block_d: tl.constexpr,
):
    # One program handles q/k/v for one [local_head, dst_rank, local_img_s].
    # The output is [local_head, W, payload_elems_per_rank], so each head keeps a
    # contiguous per-rank payload buffer for its independent all-to-all.
    row = tl.program_id(0)
    offs = tl.arange(0, block_d)
    mask = offs < hidden_dims

    rows_per_head: tl.constexpr = world_size * img_len
    local_head = row // rows_per_head
    row_in_head = row - local_head * rows_per_head
    rank = row_in_head // img_len
    s = row_in_head - rank * img_len

    src_head = local_head + rank * head_count
    q_src_s = q_img_start + s
    kv_src_s = kv_img_start + s
    q_offsets = (q_src_s * full_heads + src_head) * hidden_dims + offs
    kv_offsets = (kv_src_s * full_heads + src_head) * hidden_dims + offs

    q_vals = tl.load(q + q_offsets, mask=mask, other=0.0).to(tl.float32)
    k_vals = tl.load(k + kv_offsets, mask=mask, other=0.0).to(tl.float32)
    v_vals = tl.load(v + kv_offsets, mask=mask, other=0.0).to(tl.float32)

    q_amax = tl.maximum(tl.max(tl.abs(q_vals), axis=0), 0.001953125)
    k_amax = tl.maximum(tl.max(tl.abs(k_vals), axis=0), 0.001953125)
    v_amax = tl.maximum(tl.max(tl.abs(v_vals), axis=0), 0.001953125)
    q_scale = q_amax / 448.0
    k_scale = k_amax / 448.0
    v_scale = v_amax / 448.0

    q_row = s * 3
    k_row = q_row + 1
    v_row = q_row + 2
    head_payload_base = local_head * world_size * payload_elems_per_rank
    rank_payload_base = head_payload_base + rank * payload_elems_per_rank
    q_payload = rank_payload_base + q_row * hidden_dims + offs
    k_payload = rank_payload_base + k_row * hidden_dims + offs
    v_payload = rank_payload_base + v_row * hidden_dims + offs
    tl.store(payload_fp8 + q_payload, (q_vals / q_scale).to(tl.float8e4nv), mask=mask)
    tl.store(payload_fp8 + k_payload, (k_vals / k_scale).to(tl.float8e4nv), mask=mask)
    tl.store(payload_fp8 + v_payload, (v_vals / v_scale).to(tl.float8e4nv), mask=mask)

    head_scale_base = local_head * world_size * scale_elems_per_rank
    rank_scale_base = head_scale_base + rank * scale_elems_per_rank + scale_base_offset
    tl.store(scale_ptr + rank_scale_base + q_row, q_scale)
    tl.store(scale_ptr + rank_scale_base + k_row, k_scale)
    tl.store(scale_ptr + rank_scale_base + v_row, v_scale)


@triton.jit
def _attn_pre_fp8_kernel(
    attn,
    payload_fp8,
    scale_ptr,
    img_start,
    img_len: tl.constexpr,
    world_size: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    payload_elems_per_rank: tl.constexpr,
    scale_base_offset: tl.constexpr,
    scale_elems_per_rank: tl.constexpr,
    block_d: tl.constexpr,
):
    # One program handles one [shard_head, local_img_s] row and writes payload plus scale.
    row = tl.program_id(0)
    offs = tl.arange(0, block_d)
    mask = offs < hidden_dims

    rows_per_rank: tl.constexpr = shard_heads * img_len
    rank = row // rows_per_rank
    row_in_rank = row - rank * rows_per_rank
    s = row_in_rank % img_len
    h = row_in_rank // img_len

    src_s = img_start + rank * img_len + s
    src_offsets = src_s * (shard_heads * hidden_dims) + h * hidden_dims + offs
    vals = tl.load(attn + src_offsets, mask=mask, other=0.0).to(tl.float32)

    amax = tl.max(tl.abs(vals), axis=0)
    amax = tl.maximum(amax, 0.001953125)
    scale = amax / 448.0
    quant = (vals / scale).to(tl.float8e4nv)

    payload_offsets = rank * payload_elems_per_rank + row_in_rank * hidden_dims + offs
    tl.store(payload_fp8 + payload_offsets, quant, mask=mask)

    scale_offset = rank * scale_elems_per_rank + scale_base_offset + row_in_rank
    tl.store(scale_ptr + scale_offset, scale)


@triton.jit
def _qkv_post_fp8_kernel(
    payload_fp8,
    scale_ptr,
    q_full,
    k_full,
    v_full,
    out_q,
    out_k,
    out_v,
    total_rows,
    txt_start,
    txt_len,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    full_heads: tl.constexpr,
    head_offset: tl.constexpr,
    head_stride: tl.constexpr,
    payload_elems_per_rank: tl.constexpr,
    scale_base_offset: tl.constexpr,
    scale_elems_per_rank: tl.constexpr,
    img_first: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_d)
    row_mask = rows < total_rows
    col_mask = cols < hidden_dims
    mask = row_mask[:, None] & col_mask[None, :]

    # Tile layout: rows index [global_or_text_s, local_shard_head], cols index hidden.
    # Each image row reads one fp8 payload row and one fp32 scale; scale is broadcast
    # across hidden instead of being reloaded for every element.
    h = rows % shard_heads
    out_s = rows // shard_heads

    final_len = total_rows // shard_heads
    global_img_len = final_len - txt_len
    local_img_len = global_img_len // world_size
    if img_first:
        is_img = out_s < global_img_len
        img_global_s = out_s
        txt_s = out_s - global_img_len
    else:
        is_img = out_s >= txt_len
        img_global_s = out_s - txt_len
        txt_s = out_s

    img_rank = img_global_s // local_img_len
    img_s = img_global_s - img_rank * local_img_len
    q_row = img_s * 3 * shard_heads + h
    k_row = q_row + shard_heads
    v_row = q_row + 2 * shard_heads

    q_payload = img_rank[:, None] * payload_elems_per_rank + q_row[:, None] * hidden_dims + cols[None, :]
    k_payload = img_rank[:, None] * payload_elems_per_rank + k_row[:, None] * hidden_dims + cols[None, :]
    v_payload = img_rank[:, None] * payload_elems_per_rank + v_row[:, None] * hidden_dims + cols[None, :]
    q_scale_offset = img_rank * scale_elems_per_rank + scale_base_offset + q_row
    k_scale_offset = img_rank * scale_elems_per_rank + scale_base_offset + k_row
    v_scale_offset = img_rank * scale_elems_per_rank + scale_base_offset + v_row

    img_mask = mask & is_img[:, None]
    q_scale = tl.load(scale_ptr + q_scale_offset, mask=row_mask & is_img, other=0.0).to(tl.float32)[:, None]
    k_scale = tl.load(scale_ptr + k_scale_offset, mask=row_mask & is_img, other=0.0).to(tl.float32)[:, None]
    v_scale = tl.load(scale_ptr + v_scale_offset, mask=row_mask & is_img, other=0.0).to(tl.float32)[:, None]
    q_img = tl.load(payload_fp8 + q_payload, mask=img_mask, other=0.0).to(tl.float32) * q_scale
    k_img = tl.load(payload_fp8 + k_payload, mask=img_mask, other=0.0).to(tl.float32) * k_scale
    v_img = tl.load(payload_fp8 + v_payload, mask=img_mask, other=0.0).to(tl.float32) * v_scale

    txt_head = head_offset + cur_rank * head_stride + h
    txt_src_s = txt_start + txt_s
    txt_base = (txt_src_s[:, None] * full_heads + txt_head[:, None]) * hidden_dims + cols[None, :]
    txt_mask = mask & ~is_img[:, None]
    q_txt = tl.load(q_full + txt_base, mask=txt_mask, other=0.0).to(tl.float32)
    k_txt = tl.load(k_full + txt_base, mask=txt_mask, other=0.0).to(tl.float32)
    v_txt = tl.load(v_full + txt_base, mask=txt_mask, other=0.0).to(tl.float32)

    out_offsets = rows[:, None] * hidden_dims + cols[None, :]
    tl.store(out_q + out_offsets, q_img + q_txt, mask=mask)
    tl.store(out_k + out_offsets, k_img + k_txt, mask=mask)
    tl.store(out_v + out_offsets, v_img + v_txt, mask=mask)


@triton.jit
def _qonly_qkv_post_fp8_kernel(
    payload_fp8,
    scale_ptr,
    k_full,
    v_full,
    out_q,
    out_k,
    out_v,
    total_rows,
    q_rows,
    txt_start,
    txt_len,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    full_heads: tl.constexpr,
    head_offset: tl.constexpr,
    head_stride: tl.constexpr,
    payload_elems_per_rank: tl.constexpr,
    scale_base_offset: tl.constexpr,
    scale_elems_per_rank: tl.constexpr,
    img_first: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_d)
    row_mask = rows < total_rows
    col_mask = cols < hidden_dims
    mask = row_mask[:, None] & col_mask[None, :]

    h = rows % shard_heads
    out_s = rows // shard_heads

    global_img_len = q_rows // shard_heads
    local_img_len = global_img_len // world_size
    kv_final_len = global_img_len + txt_len
    kv_rows = kv_final_len * shard_heads

    is_q_row = rows < q_rows
    q_img_rank = out_s // local_img_len
    q_img_s = out_s - q_img_rank * local_img_len
    q_row = q_img_s * 3 * shard_heads + h
    q_payload = q_img_rank[:, None] * payload_elems_per_rank + q_row[:, None] * hidden_dims + cols[None, :]
    q_scale_offset = q_img_rank * scale_elems_per_rank + scale_base_offset + q_row
    q_scale = tl.load(scale_ptr + q_scale_offset, mask=row_mask & is_q_row, other=0.0).to(tl.float32)[:, None]
    q_val = tl.load(payload_fp8 + q_payload, mask=mask & is_q_row[:, None], other=0.0).to(tl.float32) * q_scale
    tl.store(out_q + rows[:, None] * hidden_dims + cols[None, :], q_val, mask=mask & is_q_row[:, None])

    is_kv_row = rows < kv_rows
    if img_first:
        is_img = out_s < global_img_len
        img_global_s = out_s
        txt_s = out_s - global_img_len
    else:
        is_img = out_s >= txt_len
        img_global_s = out_s - txt_len
        txt_s = out_s

    img_rank = img_global_s // local_img_len
    img_s = img_global_s - img_rank * local_img_len
    qkv_row = img_s * 3 * shard_heads + h
    k_row = qkv_row + shard_heads
    v_row = qkv_row + 2 * shard_heads
    k_payload = img_rank[:, None] * payload_elems_per_rank + k_row[:, None] * hidden_dims + cols[None, :]
    v_payload = img_rank[:, None] * payload_elems_per_rank + v_row[:, None] * hidden_dims + cols[None, :]
    k_scale_offset = img_rank * scale_elems_per_rank + scale_base_offset + k_row
    v_scale_offset = img_rank * scale_elems_per_rank + scale_base_offset + v_row
    img_mask = mask & is_kv_row[:, None] & is_img[:, None]
    k_scale = tl.load(scale_ptr + k_scale_offset, mask=row_mask & is_kv_row & is_img, other=0.0).to(tl.float32)[:, None]
    v_scale = tl.load(scale_ptr + v_scale_offset, mask=row_mask & is_kv_row & is_img, other=0.0).to(tl.float32)[:, None]
    k_img = tl.load(payload_fp8 + k_payload, mask=img_mask, other=0.0).to(tl.float32) * k_scale
    v_img = tl.load(payload_fp8 + v_payload, mask=img_mask, other=0.0).to(tl.float32) * v_scale

    txt_head = head_offset + cur_rank * head_stride + h
    txt_src_s = txt_start + txt_s
    txt_base = (txt_src_s[:, None] * full_heads + txt_head[:, None]) * hidden_dims + cols[None, :]
    txt_mask = mask & is_kv_row[:, None] & ~is_img[:, None]
    k_txt = tl.load(k_full + txt_base, mask=txt_mask, other=0.0).to(tl.float32)
    v_txt = tl.load(v_full + txt_base, mask=txt_mask, other=0.0).to(tl.float32)

    kv_offsets = rows[:, None] * hidden_dims + cols[None, :]
    kv_mask = mask & is_kv_row[:, None]
    tl.store(out_k + kv_offsets, k_img + k_txt, mask=kv_mask)
    tl.store(out_v + kv_offsets, v_img + v_txt, mask=kv_mask)


@triton.jit
def _attn_post_fp8_kernel(
    payload_fp8,
    scale_ptr,
    txt_attn,
    out,
    total_rows,
    txt_len,
    world_size: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    payload_elems_per_rank: tl.constexpr,
    scale_base_offset: tl.constexpr,
    scale_elems_per_rank: tl.constexpr,
    img_first: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_d)
    row_mask = rows < total_rows
    col_mask = cols < hidden_dims
    mask = row_mask[:, None] & col_mask[None, :]

    # Tile layout: rows index [local_img_or_text_s, rank, shard_head], cols index hidden.
    # Image rows are dequantized from split payload/scale tensors; text rows are already
    # gathered in full-head order and copied through this same post kernel.
    hidden_rows_per_seq = world_size * shard_heads
    h = rows % shard_heads
    tmp = rows // shard_heads
    rank = tmp % world_size
    out_s = tmp // world_size

    final_len = total_rows // hidden_rows_per_seq
    img_len = final_len - txt_len
    if img_first:
        is_img = out_s < img_len
        img_s = out_s
        txt_s = out_s - img_len
    else:
        is_img = out_s >= txt_len
        img_s = out_s - txt_len
        txt_s = out_s

    row_in_rank = h * img_len + img_s
    payload_offset = rank[:, None] * payload_elems_per_rank + row_in_rank[:, None] * hidden_dims + cols[None, :]
    scale_offset = rank * scale_elems_per_rank + scale_base_offset + row_in_rank
    img_scale = tl.load(scale_ptr + scale_offset, mask=row_mask & is_img, other=0.0).to(tl.float32)[:, None]
    img_mask = mask & is_img[:, None]
    img_val = tl.load(payload_fp8 + payload_offset, mask=img_mask, other=0.0).to(tl.float32) * img_scale

    all_hidden = world_size * shard_heads * hidden_dims
    txt_base = txt_s[:, None] * all_hidden + (rank[:, None] * shard_heads + h[:, None]) * hidden_dims + cols[None, :]
    txt_val = tl.load(txt_attn + txt_base, mask=mask & ~is_img[:, None], other=0.0).to(tl.float32)

    out_offsets = rows[:, None] * hidden_dims + cols[None, :]
    tl.store(out + out_offsets, img_val + txt_val, mask=mask)


_LEGACY_ULYSSES_HINT = "Set parallel.seq_p_attn_type='ulysses' to use the legacy Ulysses implementation for this model."


def _unsupported(message):
    raise NotImplementedError(f"ulysses-opt {message}. {_LEGACY_ULYSSES_HINT}")


def _check_cuda_contiguous(*tensors):
    for tensor in tensors:
        if not tensor.is_cuda:
            _unsupported(f"layout kernels require CUDA tensors, got device={tensor.device}")
        if not tensor.is_contiguous():
            _unsupported("layout kernels require contiguous tensors")


def _check_hidden_dims(hidden_dims, path_name):
    if hidden_dims > 1024:
        _unsupported(f"{path_name} currently supports hidden_dims <= 1024, got hidden_dims={hidden_dims}")


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


def _resolve_block_m(block_size, hidden_dims, default):
    """Interpret public block_size while keeping old benchmark calls usable.

    Older microbenchmarks passed element counts such as 1024. The fused row-tiled
    kernels now use block_m as a row count, so large values are converted to an
    approximate row count while small values are treated as explicit block_m.
    """
    if block_size is None:
        return default
    if block_size >= 256:
        return max(1, int(block_size) // int(hidden_dims))
    return int(block_size)


def _numel_from_shape(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _fp8_split_layout(payload_shape, scale_shape, device, scale_dtype=torch.float32, name="fp8 split"):
    """Return per-rank split metadata for separate FP8 payload/scale tensors."""
    if int(payload_shape[0]) != int(scale_shape[0]):
        raise ValueError(f"{name}: payload and scale must share the all-to-all split dimension.")
    payload_elems_per_rank = _numel_from_shape(payload_shape[1:])
    scale_elems_per_rank = _numel_from_shape(scale_shape[1:])
    scale_item_size = torch.empty((), dtype=scale_dtype, device=device).element_size()
    return payload_elems_per_rank, scale_elems_per_rank, scale_item_size


def _check_fp8_split(payload, scale, payload_shape, scale_shape, scale_dtype=torch.float32, name="fp8 split"):
    """Validate separate FP8 payload and FP32 scale tensors used by post kernels."""
    _check_cuda_contiguous(payload, scale)
    if payload.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{name}: payload dtype must be torch.float8_e4m3fn, got {payload.dtype}.")
    if scale.dtype != scale_dtype:
        raise ValueError(f"{name}: scale dtype must be {scale_dtype}, got {scale.dtype}.")
    if tuple(payload.shape) != tuple(payload_shape):
        raise ValueError(f"{name}: payload shape {tuple(payload.shape)} does not match expected {tuple(payload_shape)}.")
    if tuple(scale.shape) != tuple(scale_shape):
        raise ValueError(f"{name}: scale shape {tuple(scale.shape)} does not match expected {tuple(scale_shape)}.")
    return _fp8_split_layout(payload_shape, scale_shape, payload.device, scale_dtype, name)


def qkv_pre(q, k, v, img_start, img_len, world_size, block_size=None):
    """Prepare BF16/FP16 q/k/v for the first Ulysses all-to-all.

    Inputs are this rank's full local sequence tensors ``[S_local, H, D]``.
    Only image rows ``img_start : img_start + img_len`` are communicated.
    The output is rank-major ``[W, S_img_local, 3, H/W, D]`` so each
    all-to-all split is already contiguous for one destination rank.
    """
    _check_cuda_contiguous(q, k, v)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v must have the same shape for qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    _, heads, hidden_dims = q.shape
    if heads % world_size != 0:
        raise ValueError(f"heads={heads} must be divisible by world_size={world_size}.")
    _check_hidden_dims(hidden_dims, "fused QKV pre path")

    shard_heads = heads // world_size
    out = torch.empty((world_size, img_len, 3, shard_heads, hidden_dims), device=q.device, dtype=q.dtype)
    total_rows = world_size * img_len * shard_heads
    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qkv_pre_kernel[grid](
        q,
        k,
        v,
        out,
        total_rows,
        img_start,
        img_start,
        img_len,
        world_size,
        shard_heads,
        hidden_dims,
        heads,
        0,
        shard_heads,
        block_m,
        block_d,
        num_warps=4,
    )
    return out


def qonly_qkv_pre(q, k, v, kv_img_start, img_len, world_size, block_size=None):
    """Prepare q_only_img q/k/v for the first Ulysses all-to-all.

    q is image-only ``[S_img, H, D]`` while k/v are full local sequence
    tensors. The communicated layout is still ``[W, S_img, 3, H/W, D]``.
    """
    _check_cuda_contiguous(q, k, v)
    _, heads, hidden_dims = q.shape
    if q.shape[0] != img_len:
        raise ValueError(f"q_only q length must equal img_len={img_len}, got q_len={q.shape[0]}.")
    if k.shape[1:] != q.shape[1:] or v.shape[1:] != q.shape[1:]:
        raise ValueError(f"q/k/v head dimensions must match for q_only qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    if heads % world_size != 0:
        raise ValueError(f"heads={heads} must be divisible by world_size={world_size}.")
    _check_hidden_dims(hidden_dims, "q_only QKV pre path")

    shard_heads = heads // world_size
    out = torch.empty((world_size, img_len, 3, shard_heads, hidden_dims), device=q.device, dtype=q.dtype)
    total_rows = world_size * img_len * shard_heads
    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qkv_pre_kernel[grid](
        q,
        k,
        v,
        out,
        total_rows,
        0,
        kv_img_start,
        img_len,
        world_size,
        shard_heads,
        hidden_dims,
        heads,
        0,
        shard_heads,
        block_m,
        block_d,
        num_warps=4,
    )
    return out


def qkv_pre_head(q, k, v, img_start, img_len, world_size, head_offset, head_stride, block_size=None):
    """Head-parallel qkv_pre without materializing ``q[:, head::stride]``.

    The output keeps the normal all-to-all contract but has one shard head:
    ``[W, S_img, 3, 1, D]``. Source heads are read from the original full-head
    tensors as ``head_offset + dst_rank * head_stride``.
    """
    _check_cuda_contiguous(q, k, v)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v must have the same shape for head qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    _, heads, hidden_dims = q.shape
    if not (0 <= head_offset < head_stride):
        raise ValueError(f"head_offset must identify one local head lane, got head_offset={head_offset}, head_stride={head_stride}.")
    if head_offset + (world_size - 1) * head_stride >= heads:
        raise ValueError(f"head lane is out of range for q/k/v heads: head_offset={head_offset}, head_stride={head_stride}, world_size={world_size}, heads={heads}.")
    _check_hidden_dims(hidden_dims, "head QKV pre path")

    shard_heads = 1
    out = torch.empty((world_size, img_len, 3, shard_heads, hidden_dims), device=q.device, dtype=q.dtype)
    total_rows = world_size * img_len * shard_heads
    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qkv_pre_kernel[grid](
        q,
        k,
        v,
        out,
        total_rows,
        img_start,
        img_start,
        img_len,
        world_size,
        shard_heads,
        hidden_dims,
        heads,
        head_offset,
        head_stride,
        block_m,
        block_d,
        num_warps=4,
    )
    return out


def qonly_qkv_pre_head(q, k, v, kv_img_start, img_len, world_size, head_offset, head_stride, block_size=None):
    """q_only_img variant of qkv_pre_head."""
    _check_cuda_contiguous(q, k, v)
    _, heads, hidden_dims = q.shape
    if q.shape[0] != img_len:
        raise ValueError(f"q_only q length must equal img_len={img_len}, got q_len={q.shape[0]}.")
    if k.shape[1:] != q.shape[1:] or v.shape[1:] != q.shape[1:]:
        raise ValueError(f"q/k/v head dimensions must match for q_only head qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    if not (0 <= head_offset < head_stride):
        raise ValueError(f"head_offset must identify one local head lane, got head_offset={head_offset}, head_stride={head_stride}.")
    if head_offset + (world_size - 1) * head_stride >= heads:
        raise ValueError(f"head lane is out of range for q/k/v heads: head_offset={head_offset}, head_stride={head_stride}, world_size={world_size}, heads={heads}.")
    _check_hidden_dims(hidden_dims, "q_only head QKV pre path")

    shard_heads = 1
    out = torch.empty((world_size, img_len, 3, shard_heads, hidden_dims), device=q.device, dtype=q.dtype)
    total_rows = world_size * img_len * shard_heads
    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qkv_pre_kernel[grid](
        q,
        k,
        v,
        out,
        total_rows,
        0,
        kv_img_start,
        img_len,
        world_size,
        shard_heads,
        hidden_dims,
        heads,
        head_offset,
        head_stride,
        block_m,
        block_d,
        num_warps=4,
    )
    return out


def qkv_pre_heads(q, k, v, img_start, img_len, world_size, head_count, block_size=None):
    """Batched head-parallel qkv_pre.

    Returns ``[H_local, W, S_img, 3, 1, D]``. Each ``out[h]`` is contiguous
    and keeps the exact ``qkv_pre_head`` all-to-all contract, while one Triton
    launch fills every local-head buffer.
    """
    _check_cuda_contiguous(q, k, v)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v must have the same shape for head qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    _, heads, hidden_dims = q.shape
    if int(head_count) <= 0:
        raise ValueError(f"head_count must be positive, got head_count={head_count}.")
    if int(head_count) * int(world_size) != heads:
        raise ValueError(f"head_count must equal heads // world_size, got head_count={head_count}, world_size={world_size}, heads={heads}.")
    _check_hidden_dims(hidden_dims, "batched head QKV pre path")

    out = torch.empty((head_count, world_size, img_len, 3, 1, hidden_dims), device=q.device, dtype=q.dtype)
    total_rows = head_count * world_size * img_len
    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qkv_pre_heads_kernel[grid](
        q,
        k,
        v,
        out,
        total_rows,
        img_start,
        img_start,
        img_len,
        world_size,
        head_count,
        hidden_dims,
        heads,
        block_m,
        block_d,
        num_warps=4,
    )
    return out


def qonly_qkv_pre_heads(q, k, v, kv_img_start, img_len, world_size, head_count, block_size=None):
    """q_only_img batched head-parallel qkv_pre."""
    _check_cuda_contiguous(q, k, v)
    _, heads, hidden_dims = q.shape
    if q.shape[0] != img_len:
        raise ValueError(f"q_only q length must equal img_len={img_len}, got q_len={q.shape[0]}.")
    if k.shape[1:] != q.shape[1:] or v.shape[1:] != q.shape[1:]:
        raise ValueError(f"q/k/v head dimensions must match for q_only head qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    if int(head_count) <= 0:
        raise ValueError(f"head_count must be positive, got head_count={head_count}.")
    if int(head_count) * int(world_size) != heads:
        raise ValueError(f"head_count must equal heads // world_size, got head_count={head_count}, world_size={world_size}, heads={heads}.")
    _check_hidden_dims(hidden_dims, "q_only batched head QKV pre path")

    out = torch.empty((head_count, world_size, img_len, 3, 1, hidden_dims), device=q.device, dtype=q.dtype)
    total_rows = head_count * world_size * img_len
    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qkv_pre_heads_kernel[grid](
        q,
        k,
        v,
        out,
        total_rows,
        0,
        kv_img_start,
        img_len,
        world_size,
        head_count,
        hidden_dims,
        heads,
        block_m,
        block_d,
        num_warps=4,
    )
    return out


def qkv_post(qkv, q_full, k_full, v_full, cur_rank, txt_start, txt_len, img_first, block_size=None):
    """Finish the first all-to-all and produce local-head q/k/v for attention.

    ``qkv`` is the communicated image tensor ``[W, S_img_local, 3, H/W, D]``.
    The kernel expands image rows to ``[W * S_img_local, H/W, D]`` and
    splices this rank's local text heads from the original full-head q/k/v.
    """
    _check_cuda_contiguous(qkv, q_full, k_full, v_full)
    world_size, img_len, qkv_count, shard_heads, hidden_dims = qkv.shape
    if qkv_count != 3:
        raise ValueError(f"qkv fused buffer must have qkv dimension == 3, got qkv_count={qkv_count}.")
    _check_hidden_dims(hidden_dims, "fused QKV post path")

    final_len = world_size * img_len + txt_len
    shape = (final_len, shard_heads, hidden_dims)
    out_q = torch.empty(shape, device=qkv.device, dtype=qkv.dtype)
    out_k = torch.empty_like(out_q)
    out_v = torch.empty_like(out_q)
    total_rows = final_len * shard_heads
    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qkv_post_kernel[grid](
        qkv,
        q_full,
        k_full,
        v_full,
        out_q,
        out_k,
        out_v,
        total_rows,
        txt_start,
        txt_len,
        cur_rank,
        world_size,
        shard_heads,
        hidden_dims,
        q_full.shape[1],
        0,
        shard_heads,
        img_first,
        block_m,
        block_d,
        num_warps=4,
    )
    return out_q, out_k, out_v


def qonly_qkv_post(qkv, k_full, v_full, cur_rank, txt_start, txt_len, img_first, block_size=None):
    """Finish q_only_img first all-to-all.

    Returns image-only q ``[W*S_img, H/W, D]`` and full k/v
    ``[W*S_img + S_txt, H/W, D]``.
    """
    _check_cuda_contiguous(qkv, k_full, v_full)
    world_size, img_len, qkv_count, shard_heads, hidden_dims = qkv.shape
    if qkv_count != 3:
        raise ValueError(f"qkv fused buffer must have qkv dimension == 3, got qkv_count={qkv_count}.")
    _check_hidden_dims(hidden_dims, "q_only QKV post path")

    q_shape = (world_size * img_len, shard_heads, hidden_dims)
    kv_shape = (world_size * img_len + txt_len, shard_heads, hidden_dims)
    out_q = torch.empty(q_shape, device=qkv.device, dtype=qkv.dtype)
    out_k = torch.empty(kv_shape, device=qkv.device, dtype=qkv.dtype)
    out_v = torch.empty_like(out_k)
    q_rows = q_shape[0] * shard_heads
    kv_rows = kv_shape[0] * shard_heads
    total_rows = max(q_rows, kv_rows)
    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qonly_qkv_post_kernel[grid](
        qkv,
        k_full,
        v_full,
        out_q,
        out_k,
        out_v,
        total_rows,
        q_rows,
        txt_start,
        txt_len,
        cur_rank,
        world_size,
        shard_heads,
        hidden_dims,
        k_full.shape[1],
        0,
        shard_heads,
        img_first,
        block_m,
        block_d,
        num_warps=4,
    )
    return out_q, out_k, out_v


def qkv_post_head(qkv, q_full, k_full, v_full, cur_rank, txt_start, txt_len, img_first, head_offset, head_stride, block_size=None):
    """Head-parallel qkv_post that splices text from full-head q/k/v."""
    _check_cuda_contiguous(qkv, q_full, k_full, v_full)
    world_size, img_len, qkv_count, shard_heads, hidden_dims = qkv.shape
    if qkv_count != 3 or shard_heads != 1:
        raise ValueError(f"head qkv buffer must have shape [W, S_img, 3, 1, D], got qkv_count={qkv_count}, shard_heads={shard_heads}.")
    if q_full.shape != k_full.shape or q_full.shape != v_full.shape:
        raise ValueError(f"q/k/v full tensors must have the same shape, got q={tuple(q_full.shape)}, k={tuple(k_full.shape)}, v={tuple(v_full.shape)}.")
    if head_offset + (world_size - 1) * head_stride >= q_full.shape[1]:
        raise ValueError(f"head lane is out of range for q/k/v heads: head_offset={head_offset}, head_stride={head_stride}, world_size={world_size}, heads={q_full.shape[1]}.")
    _check_hidden_dims(hidden_dims, "head QKV post path")

    final_len = world_size * img_len + txt_len
    shape = (final_len, shard_heads, hidden_dims)
    out_q = torch.empty(shape, device=qkv.device, dtype=qkv.dtype)
    out_k = torch.empty_like(out_q)
    out_v = torch.empty_like(out_q)
    total_rows = final_len * shard_heads
    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qkv_post_kernel[grid](
        qkv,
        q_full,
        k_full,
        v_full,
        out_q,
        out_k,
        out_v,
        total_rows,
        txt_start,
        txt_len,
        cur_rank,
        world_size,
        shard_heads,
        hidden_dims,
        q_full.shape[1],
        head_offset,
        head_stride,
        img_first,
        block_m,
        block_d,
        num_warps=4,
    )
    return out_q, out_k, out_v


def qonly_qkv_post_head(qkv, k_full, v_full, cur_rank, txt_start, txt_len, img_first, head_offset, head_stride, block_size=None):
    """q_only_img variant of qkv_post_head."""
    _check_cuda_contiguous(qkv, k_full, v_full)
    world_size, img_len, qkv_count, shard_heads, hidden_dims = qkv.shape
    if qkv_count != 3 or shard_heads != 1:
        raise ValueError(f"head qkv buffer must have shape [W, S_img, 3, 1, D], got qkv_count={qkv_count}, shard_heads={shard_heads}.")
    if k_full.shape != v_full.shape:
        raise ValueError(f"k/v full tensors must have the same shape, got k={tuple(k_full.shape)}, v={tuple(v_full.shape)}.")
    if head_offset + (world_size - 1) * head_stride >= k_full.shape[1]:
        raise ValueError(f"head lane is out of range for k/v heads: head_offset={head_offset}, head_stride={head_stride}, world_size={world_size}, heads={k_full.shape[1]}.")
    _check_hidden_dims(hidden_dims, "q_only head QKV post path")

    q_shape = (world_size * img_len, shard_heads, hidden_dims)
    kv_shape = (world_size * img_len + txt_len, shard_heads, hidden_dims)
    out_q = torch.empty(q_shape, device=qkv.device, dtype=qkv.dtype)
    out_k = torch.empty(kv_shape, device=qkv.device, dtype=qkv.dtype)
    out_v = torch.empty_like(out_k)
    q_rows = q_shape[0] * shard_heads
    kv_rows = kv_shape[0] * shard_heads
    total_rows = max(q_rows, kv_rows)
    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qonly_qkv_post_kernel[grid](
        qkv,
        k_full,
        v_full,
        out_q,
        out_k,
        out_v,
        total_rows,
        q_rows,
        txt_start,
        txt_len,
        cur_rank,
        world_size,
        shard_heads,
        hidden_dims,
        k_full.shape[1],
        head_offset,
        head_stride,
        img_first,
        block_m,
        block_d,
        num_warps=4,
    )
    return out_q, out_k, out_v


def attn_pre(attn, img_start, img_len, world_size, shard_heads, hidden_dims, block_size=None):
    """Prepare BF16/FP16 attention output for the second all-to-all.

    ``attn`` is the local-head attention result ``[S_global_img + S_txt, H/W*D]``.
    Image rows are grouped by source rank and emitted as ``[W, H/W, S_img_local, D]``.
    """
    _check_cuda_contiguous(attn)
    _check_hidden_dims(hidden_dims, "fused attn pre path")

    out = torch.empty((world_size, shard_heads, img_len, hidden_dims), device=attn.device, dtype=attn.dtype)
    total_rows = world_size * shard_heads * img_len
    block_m = _resolve_block_m(block_size, hidden_dims, default=16)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _attn_pre_kernel[grid](
        attn,
        out,
        total_rows,
        img_start,
        img_len,
        world_size,
        shard_heads,
        hidden_dims,
        block_m,
        block_d,
        num_warps=4,
    )
    return out


def attn_post(attn, txt_attn, img_first, block_size=None):
    """Finish the second all-to-all and return the full-head output sequence.

    ``attn`` carries communicated image rows ``[W, H/W, S_img_local, D]``.
    ``txt_attn`` is already all-gathered across head shards. The result is
    ``[S_img_local + S_txt, H*D]`` in the configured image/text order.
    """
    _check_cuda_contiguous(attn, txt_attn)
    world_size, shard_heads, img_len, hidden_dims = attn.shape
    _check_hidden_dims(hidden_dims, "fused attn post path")

    txt_len = txt_attn.shape[0]
    final_len = img_len + txt_len
    out = torch.empty((final_len, world_size * shard_heads * hidden_dims), device=attn.device, dtype=attn.dtype)
    total_rows = final_len * world_size * shard_heads
    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _attn_post_kernel[grid](
        attn,
        txt_attn,
        out,
        total_rows,
        txt_len,
        world_size,
        shard_heads,
        hidden_dims,
        img_first,
        block_m,
        block_d,
        num_warps=4,
    )
    return out


def qkv_pre_fp8(q, k, v, img_start, img_len, world_size):
    """FP8 version of ``qkv_pre`` with fused layout+quantization.

    Returns separate payload/scale tensors so runtime communication can keep
    the faster legacy NCCL shape: one all-to-all for FP8 payload and one for
    FP32 per-row scale.
    """
    _check_cuda_contiguous(q, k, v)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v must have the same shape for qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    _, heads, hidden_dims = q.shape
    if heads % world_size != 0:
        raise ValueError(f"heads={heads} must be divisible by world_size={world_size}.")
    _check_hidden_dims(hidden_dims, "fused QKV fp8 pre path")

    shard_heads = heads // world_size
    payload_shape = (world_size, img_len, 3, shard_heads, hidden_dims)
    scale_shape = (*payload_shape[:-1], 1)
    payload_elems_per_rank, scale_elems_per_rank, _scale_item_size = _fp8_split_layout(payload_shape, scale_shape, q.device, name="fused QKV fp8 split path")

    payload = torch.empty(payload_shape, device=q.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(scale_shape, device=q.device, dtype=torch.float32)
    total_rows = world_size * img_len * shard_heads
    block_d = _next_power_of_2(hidden_dims)
    _qkv_pre_fp8_kernel[(total_rows,)](
        q,
        k,
        v,
        payload,
        scale,
        img_start,
        img_start,
        img_len,
        world_size,
        shard_heads,
        hidden_dims,
        heads,
        0,
        shard_heads,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        block_d,
        num_warps=4,
    )
    return payload, scale, payload_shape, scale_shape


def qonly_qkv_pre_fp8(q, k, v, kv_img_start, img_len, world_size):
    """FP8 q_only_img pre path with fused layout+quantization."""
    _check_cuda_contiguous(q, k, v)
    _, heads, hidden_dims = q.shape
    if q.shape[0] != img_len:
        raise ValueError(f"q_only q length must equal img_len={img_len}, got q_len={q.shape[0]}.")
    if k.shape[1:] != q.shape[1:] or v.shape[1:] != q.shape[1:]:
        raise ValueError(f"q/k/v head dimensions must match for q_only qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    if heads % world_size != 0:
        raise ValueError(f"heads={heads} must be divisible by world_size={world_size}.")
    _check_hidden_dims(hidden_dims, "q_only QKV fp8 pre path")

    shard_heads = heads // world_size
    payload_shape = (world_size, img_len, 3, shard_heads, hidden_dims)
    scale_shape = (*payload_shape[:-1], 1)
    payload_elems_per_rank, scale_elems_per_rank, _scale_item_size = _fp8_split_layout(payload_shape, scale_shape, q.device, name="q_only QKV fp8 split path")

    payload = torch.empty(payload_shape, device=q.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(scale_shape, device=q.device, dtype=torch.float32)
    total_rows = world_size * img_len * shard_heads
    block_d = _next_power_of_2(hidden_dims)
    _qkv_pre_fp8_kernel[(total_rows,)](
        q,
        k,
        v,
        payload,
        scale,
        0,
        kv_img_start,
        img_len,
        world_size,
        shard_heads,
        hidden_dims,
        heads,
        0,
        shard_heads,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        block_d,
        num_warps=4,
    )
    return payload, scale, payload_shape, scale_shape


def qkv_pre_head_fp8(q, k, v, img_start, img_len, world_size, head_offset, head_stride):
    """FP8 qkv_pre_head with fused quantization and split payload/scale output."""
    _check_cuda_contiguous(q, k, v)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v must have the same shape for head qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    _, heads, hidden_dims = q.shape
    if not (0 <= head_offset < head_stride):
        raise ValueError(f"head_offset must identify one local head lane, got head_offset={head_offset}, head_stride={head_stride}.")
    if head_offset + (world_size - 1) * head_stride >= heads:
        raise ValueError(f"head lane is out of range for q/k/v heads: head_offset={head_offset}, head_stride={head_stride}, world_size={world_size}, heads={heads}.")
    _check_hidden_dims(hidden_dims, "head QKV fp8 pre path")

    shard_heads = 1
    payload_shape = (world_size, img_len, 3, shard_heads, hidden_dims)
    scale_shape = (*payload_shape[:-1], 1)
    payload_elems_per_rank, scale_elems_per_rank, _scale_item_size = _fp8_split_layout(payload_shape, scale_shape, q.device, name="head QKV fp8 split path")

    payload = torch.empty(payload_shape, device=q.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(scale_shape, device=q.device, dtype=torch.float32)
    total_rows = world_size * img_len * shard_heads
    block_d = _next_power_of_2(hidden_dims)
    _qkv_pre_fp8_kernel[(total_rows,)](
        q,
        k,
        v,
        payload,
        scale,
        img_start,
        img_start,
        img_len,
        world_size,
        shard_heads,
        hidden_dims,
        heads,
        head_offset,
        head_stride,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        block_d,
        num_warps=4,
    )
    return payload, scale, payload_shape, scale_shape


def qonly_qkv_pre_head_fp8(q, k, v, kv_img_start, img_len, world_size, head_offset, head_stride):
    """q_only_img FP8 variant of qkv_pre_head_fp8."""
    _check_cuda_contiguous(q, k, v)
    _, heads, hidden_dims = q.shape
    if q.shape[0] != img_len:
        raise ValueError(f"q_only q length must equal img_len={img_len}, got q_len={q.shape[0]}.")
    if k.shape[1:] != q.shape[1:] or v.shape[1:] != q.shape[1:]:
        raise ValueError(f"q/k/v head dimensions must match for q_only head qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    if not (0 <= head_offset < head_stride):
        raise ValueError(f"head_offset must identify one local head lane, got head_offset={head_offset}, head_stride={head_stride}.")
    if head_offset + (world_size - 1) * head_stride >= heads:
        raise ValueError(f"head lane is out of range for q/k/v heads: head_offset={head_offset}, head_stride={head_stride}, world_size={world_size}, heads={heads}.")
    _check_hidden_dims(hidden_dims, "q_only head QKV fp8 pre path")

    shard_heads = 1
    payload_shape = (world_size, img_len, 3, shard_heads, hidden_dims)
    scale_shape = (*payload_shape[:-1], 1)
    payload_elems_per_rank, scale_elems_per_rank, _scale_item_size = _fp8_split_layout(payload_shape, scale_shape, q.device, name="q_only head QKV fp8 split path")

    payload = torch.empty(payload_shape, device=q.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(scale_shape, device=q.device, dtype=torch.float32)
    total_rows = world_size * img_len * shard_heads
    block_d = _next_power_of_2(hidden_dims)
    _qkv_pre_fp8_kernel[(total_rows,)](
        q,
        k,
        v,
        payload,
        scale,
        0,
        kv_img_start,
        img_len,
        world_size,
        shard_heads,
        hidden_dims,
        heads,
        head_offset,
        head_stride,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        block_d,
        num_warps=4,
    )
    return payload, scale, payload_shape, scale_shape


def qkv_pre_heads_fp8(q, k, v, img_start, img_len, world_size, head_count):
    """Batched head-parallel FP8 qkv_pre with split payload/scale output.

    Returns ``payload_heads`` and ``scale_heads`` whose leading dimension is
    the local head. Each ``payload_heads[h]`` and ``scale_heads[h]`` remains
    contiguous for its independent all-to-all.
    """
    _check_cuda_contiguous(q, k, v)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v must have the same shape for head qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    _, heads, hidden_dims = q.shape
    if int(head_count) <= 0:
        raise ValueError(f"head_count must be positive, got head_count={head_count}.")
    if int(head_count) * int(world_size) != heads:
        raise ValueError(f"head_count must equal heads // world_size, got head_count={head_count}, world_size={world_size}, heads={heads}.")
    _check_hidden_dims(hidden_dims, "batched head QKV fp8 pre path")

    payload_shape = (world_size, img_len, 3, 1, hidden_dims)
    scale_shape = (*payload_shape[:-1], 1)
    payload_elems_per_rank, scale_elems_per_rank, _scale_item_size = _fp8_split_layout(payload_shape, scale_shape, q.device, name="batched head QKV fp8 split path")

    payload = torch.empty((head_count, *payload_shape), device=q.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((head_count, *scale_shape), device=q.device, dtype=torch.float32)
    total_rows = head_count * world_size * img_len
    block_d = _next_power_of_2(hidden_dims)
    _qkv_pre_heads_fp8_kernel[(total_rows,)](
        q,
        k,
        v,
        payload,
        scale,
        img_start,
        img_start,
        img_len,
        world_size,
        head_count,
        hidden_dims,
        heads,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        block_d,
        num_warps=4,
    )
    return payload, scale, payload_shape, scale_shape


def qonly_qkv_pre_heads_fp8(q, k, v, kv_img_start, img_len, world_size, head_count):
    """q_only_img batched head-parallel FP8 qkv_pre."""
    _check_cuda_contiguous(q, k, v)
    _, heads, hidden_dims = q.shape
    if q.shape[0] != img_len:
        raise ValueError(f"q_only q length must equal img_len={img_len}, got q_len={q.shape[0]}.")
    if k.shape[1:] != q.shape[1:] or v.shape[1:] != q.shape[1:]:
        raise ValueError(f"q/k/v head dimensions must match for q_only batched qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    if int(head_count) <= 0:
        raise ValueError(f"head_count must be positive, got head_count={head_count}.")
    if int(head_count) * int(world_size) != heads:
        raise ValueError(f"head_count must equal heads // world_size, got head_count={head_count}, world_size={world_size}, heads={heads}.")
    _check_hidden_dims(hidden_dims, "q_only batched head QKV fp8 pre path")

    payload_shape = (world_size, img_len, 3, 1, hidden_dims)
    scale_shape = (*payload_shape[:-1], 1)
    payload_elems_per_rank, scale_elems_per_rank, _scale_item_size = _fp8_split_layout(payload_shape, scale_shape, q.device, name="q_only batched head QKV fp8 split path")

    payload = torch.empty((head_count, *payload_shape), device=q.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((head_count, *scale_shape), device=q.device, dtype=torch.float32)
    total_rows = head_count * world_size * img_len
    block_d = _next_power_of_2(hidden_dims)
    _qkv_pre_heads_fp8_kernel[(total_rows,)](
        q,
        k,
        v,
        payload,
        scale,
        0,
        kv_img_start,
        img_len,
        world_size,
        head_count,
        hidden_dims,
        heads,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        block_d,
        num_warps=4,
    )
    return payload, scale, payload_shape, scale_shape


def attn_pre_fp8(attn, img_start, img_len, world_size, shard_heads, hidden_dims):
    """FP8 version of ``attn_pre`` with fused layout+quantization."""
    _check_cuda_contiguous(attn)
    _check_hidden_dims(hidden_dims, "fused attn fp8 pre path")

    payload_shape = (world_size, shard_heads, img_len, hidden_dims)
    scale_shape = (*payload_shape[:-1], 1)
    payload_elems_per_rank, scale_elems_per_rank, _scale_item_size = _fp8_split_layout(payload_shape, scale_shape, attn.device, name="fused attn fp8 split path")

    payload = torch.empty(payload_shape, device=attn.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(scale_shape, device=attn.device, dtype=torch.float32)
    total_rows = world_size * shard_heads * img_len
    block_d = _next_power_of_2(hidden_dims)
    _attn_pre_fp8_kernel[(total_rows,)](
        attn,
        payload,
        scale,
        img_start,
        img_len,
        world_size,
        shard_heads,
        hidden_dims,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        block_d,
        num_warps=4,
    )
    return payload, scale, payload_shape, scale_shape


def qkv_post_fp8(payload, scale, payload_shape, scale_shape, q_full, k_full, v_full, cur_rank, txt_start, txt_len, img_first, block_size=None):
    """FP8 version of ``qkv_post`` for separate payload/scale tensors."""
    _check_cuda_contiguous(payload, scale, q_full, k_full, v_full)
    world_size, img_len, qkv_count, shard_heads, hidden_dims = payload_shape
    if qkv_count != 3:
        raise ValueError(f"qkv fused buffer must have qkv dimension == 3, got qkv_count={qkv_count}.")
    if tuple(scale_shape) != (world_size, img_len, 3, shard_heads, 1):
        raise ValueError("qkv scale shape does not match payload shape.")

    _check_hidden_dims(hidden_dims, "fused QKV fp8 post path")
    payload_elems_per_rank, scale_elems_per_rank, _scale_item_size = _check_fp8_split(payload, scale, payload_shape, scale_shape, torch.float32, name="split QKV fp8 input")

    final_len = world_size * img_len + txt_len
    shape = (final_len, shard_heads, hidden_dims)
    out_q = torch.empty(shape, device=payload.device, dtype=q_full.dtype)
    out_k = torch.empty_like(out_q)
    out_v = torch.empty_like(out_q)
    total_rows = final_len * shard_heads
    block_m = _resolve_block_m(block_size, hidden_dims, default=16)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qkv_post_fp8_kernel[grid](
        payload,
        scale,
        q_full,
        k_full,
        v_full,
        out_q,
        out_k,
        out_v,
        total_rows,
        txt_start,
        txt_len,
        cur_rank,
        world_size,
        shard_heads,
        hidden_dims,
        q_full.shape[1],
        0,
        shard_heads,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        img_first,
        block_m,
        block_d,
        num_warps=4,
    )
    return out_q, out_k, out_v


def qonly_qkv_post_fp8(payload, scale, payload_shape, scale_shape, k_full, v_full, cur_rank, txt_start, txt_len, img_first, block_size=None):
    """FP8 version of qonly_qkv_post for separate payload/scale tensors."""
    _check_cuda_contiguous(payload, scale, k_full, v_full)
    world_size, img_len, qkv_count, shard_heads, hidden_dims = payload_shape
    if qkv_count != 3:
        raise ValueError(f"qkv fused buffer must have qkv dimension == 3, got qkv_count={qkv_count}.")
    if tuple(scale_shape) != (world_size, img_len, 3, shard_heads, 1):
        raise ValueError("q_only qkv scale shape does not match payload shape.")

    _check_hidden_dims(hidden_dims, "q_only QKV fp8 post path")
    payload_elems_per_rank, scale_elems_per_rank, _scale_item_size = _check_fp8_split(payload, scale, payload_shape, scale_shape, torch.float32, name="split q_only QKV fp8 input")

    q_shape = (world_size * img_len, shard_heads, hidden_dims)
    kv_shape = (world_size * img_len + txt_len, shard_heads, hidden_dims)
    out_q = torch.empty(q_shape, device=payload.device, dtype=k_full.dtype)
    out_k = torch.empty(kv_shape, device=payload.device, dtype=k_full.dtype)
    out_v = torch.empty_like(out_k)
    q_rows = q_shape[0] * shard_heads
    kv_rows = kv_shape[0] * shard_heads
    total_rows = max(q_rows, kv_rows)
    block_m = _resolve_block_m(block_size, hidden_dims, default=16)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qonly_qkv_post_fp8_kernel[grid](
        payload,
        scale,
        k_full,
        v_full,
        out_q,
        out_k,
        out_v,
        total_rows,
        q_rows,
        txt_start,
        txt_len,
        cur_rank,
        world_size,
        shard_heads,
        hidden_dims,
        k_full.shape[1],
        0,
        shard_heads,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        img_first,
        block_m,
        block_d,
        num_warps=4,
    )
    return out_q, out_k, out_v


def qkv_post_head_fp8(payload, scale, payload_shape, scale_shape, q_full, k_full, v_full, cur_rank, txt_start, txt_len, img_first, head_offset, head_stride, block_size=None):
    """FP8 qkv_post_head for separate payload/scale tensors."""
    _check_cuda_contiguous(payload, scale, q_full, k_full, v_full)
    world_size, img_len, qkv_count, shard_heads, hidden_dims = payload_shape
    if qkv_count != 3 or shard_heads != 1:
        raise ValueError(f"head qkv payload must have shape [W, S_img, 3, 1, D], got qkv_count={qkv_count}, shard_heads={shard_heads}.")
    if tuple(scale_shape) != (world_size, img_len, 3, shard_heads, 1):
        raise ValueError("head qkv scale shape does not match payload shape.")
    if q_full.shape != k_full.shape or q_full.shape != v_full.shape:
        raise ValueError(f"q/k/v full tensors must have the same shape, got q={tuple(q_full.shape)}, k={tuple(k_full.shape)}, v={tuple(v_full.shape)}.")
    if head_offset + (world_size - 1) * head_stride >= q_full.shape[1]:
        raise ValueError(f"head lane is out of range for q/k/v heads: head_offset={head_offset}, head_stride={head_stride}, world_size={world_size}, heads={q_full.shape[1]}.")

    _check_hidden_dims(hidden_dims, "head QKV fp8 post path")
    payload_elems_per_rank, scale_elems_per_rank, _scale_item_size = _check_fp8_split(payload, scale, payload_shape, scale_shape, torch.float32, name="split head QKV fp8 input")

    final_len = world_size * img_len + txt_len
    shape = (final_len, shard_heads, hidden_dims)
    out_q = torch.empty(shape, device=payload.device, dtype=q_full.dtype)
    out_k = torch.empty_like(out_q)
    out_v = torch.empty_like(out_q)
    total_rows = final_len * shard_heads
    block_m = _resolve_block_m(block_size, hidden_dims, default=16)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qkv_post_fp8_kernel[grid](
        payload,
        scale,
        q_full,
        k_full,
        v_full,
        out_q,
        out_k,
        out_v,
        total_rows,
        txt_start,
        txt_len,
        cur_rank,
        world_size,
        shard_heads,
        hidden_dims,
        q_full.shape[1],
        head_offset,
        head_stride,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        img_first,
        block_m,
        block_d,
        num_warps=4,
    )
    return out_q, out_k, out_v


def qonly_qkv_post_head_fp8(payload, scale, payload_shape, scale_shape, k_full, v_full, cur_rank, txt_start, txt_len, img_first, head_offset, head_stride, block_size=None):
    """q_only_img FP8 variant of qkv_post_head_fp8."""
    _check_cuda_contiguous(payload, scale, k_full, v_full)
    world_size, img_len, qkv_count, shard_heads, hidden_dims = payload_shape
    if qkv_count != 3 or shard_heads != 1:
        raise ValueError(f"head qkv payload must have shape [W, S_img, 3, 1, D], got qkv_count={qkv_count}, shard_heads={shard_heads}.")
    if tuple(scale_shape) != (world_size, img_len, 3, shard_heads, 1):
        raise ValueError("q_only head qkv scale shape does not match payload shape.")
    if k_full.shape != v_full.shape:
        raise ValueError(f"k/v full tensors must have the same shape, got k={tuple(k_full.shape)}, v={tuple(v_full.shape)}.")
    if head_offset + (world_size - 1) * head_stride >= k_full.shape[1]:
        raise ValueError(f"head lane is out of range for k/v heads: head_offset={head_offset}, head_stride={head_stride}, world_size={world_size}, heads={k_full.shape[1]}.")

    _check_hidden_dims(hidden_dims, "q_only head QKV fp8 post path")
    payload_elems_per_rank, scale_elems_per_rank, _scale_item_size = _check_fp8_split(payload, scale, payload_shape, scale_shape, torch.float32, name="split q_only head QKV fp8 input")

    q_shape = (world_size * img_len, shard_heads, hidden_dims)
    kv_shape = (world_size * img_len + txt_len, shard_heads, hidden_dims)
    out_q = torch.empty(q_shape, device=payload.device, dtype=k_full.dtype)
    out_k = torch.empty(kv_shape, device=payload.device, dtype=k_full.dtype)
    out_v = torch.empty_like(out_k)
    q_rows = q_shape[0] * shard_heads
    kv_rows = kv_shape[0] * shard_heads
    total_rows = max(q_rows, kv_rows)
    block_m = _resolve_block_m(block_size, hidden_dims, default=16)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qonly_qkv_post_fp8_kernel[grid](
        payload,
        scale,
        k_full,
        v_full,
        out_q,
        out_k,
        out_v,
        total_rows,
        q_rows,
        txt_start,
        txt_len,
        cur_rank,
        world_size,
        shard_heads,
        hidden_dims,
        k_full.shape[1],
        head_offset,
        head_stride,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        img_first,
        block_m,
        block_d,
        num_warps=4,
    )
    return out_q, out_k, out_v


def attn_post_fp8(payload, scale, payload_shape, scale_shape, txt_attn, img_first, block_size=None):
    """FP8 version of ``attn_post`` for separate payload/scale tensors."""
    _check_cuda_contiguous(payload, scale, txt_attn)
    world_size, shard_heads, img_len, hidden_dims = payload_shape
    if tuple(scale_shape) != (world_size, shard_heads, img_len, 1):
        raise ValueError("attn scale shape does not match payload shape.")

    _check_hidden_dims(hidden_dims, "fused attn fp8 post path")
    payload_elems_per_rank, scale_elems_per_rank, _scale_item_size = _check_fp8_split(payload, scale, payload_shape, scale_shape, torch.float32, name="split attn fp8 input")

    txt_len = txt_attn.shape[0]
    final_len = img_len + txt_len
    out = torch.empty((final_len, world_size * shard_heads * hidden_dims), device=payload.device, dtype=txt_attn.dtype)
    total_rows = final_len * world_size * shard_heads
    block_m = _resolve_block_m(block_size, hidden_dims, default=16)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _attn_post_fp8_kernel[grid](
        payload,
        scale,
        txt_attn,
        out,
        total_rows,
        txt_len,
        world_size,
        shard_heads,
        hidden_dims,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        img_first,
        block_m,
        block_d,
        num_warps=4,
    )
    return out
