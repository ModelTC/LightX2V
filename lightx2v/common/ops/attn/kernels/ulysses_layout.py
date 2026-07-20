"""Fused layout kernels for the Ulysses Triton pre/post backend.

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

# Validation and launch-shape helpers

_TORCH_PREPOST_HINT = "Set parallel.seq_p_prepost_backend='torch' to use the unfused PyTorch pre/post backend."


def _unsupported(message):
    raise NotImplementedError(f"Ulysses Triton pre/post {message}. {_TORCH_PREPOST_HINT}")


def _check_cuda_contiguous(*tensors):
    for tensor in tensors:
        if not tensor.is_cuda:
            _unsupported(f"layout kernels require CUDA tensors, got device={tensor.device}")
        if not tensor.is_contiguous():
            _unsupported("layout kernels require contiguous tensors")


def _check_hidden_dims(hidden_dims, path_name):
    if hidden_dims > 1024:
        _unsupported(f"{path_name} currently supports hidden_dims <= 1024, got hidden_dims={hidden_dims}")


def _resolve_qkv_head_layout(heads, world_size, head_index):
    if heads % world_size != 0:
        raise ValueError(f"heads={heads} must be divisible by world_size={world_size}.")
    head_stride = heads // world_size
    if head_index is None:
        return head_stride, 0, head_stride
    if not 0 <= head_index < head_stride:
        raise ValueError(f"head_index must be in [0, {head_stride}), got {head_index}.")
    return 1, head_index, head_stride


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


def _resolve_block_m(block_size, hidden_dims, default):
    """Map legacy element-count tuning values to the row-tiled launch shape."""
    if block_size is None:
        block_m = int(default)
    elif block_size >= 256:
        block_m = max(1, int(block_size) // int(hidden_dims))
    else:
        block_m = int(block_size)
    if block_m <= 0:
        raise ValueError(f"block_size must resolve to a positive row count, got block_m={block_m}")
    return _next_power_of_2(block_m)


def _numel_from_shape(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _fp8_split_layout(payload_shape, scale_shape, name="fp8 split"):
    """Return per-rank split metadata for separate FP8 payload/scale tensors."""
    if int(payload_shape[0]) != int(scale_shape[0]):
        raise ValueError(f"{name}: payload and scale must share the all-to-all split dimension.")
    payload_elems_per_rank = _numel_from_shape(payload_shape[1:])
    scale_elems_per_rank = _numel_from_shape(scale_shape[1:])
    return payload_elems_per_rank, scale_elems_per_rank


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
    return _fp8_split_layout(payload_shape, scale_shape, name)


# BF16/FP16 layout kernels

@triton.jit
def _qkv_pre_kernel(
    q,
    k,
    v,
    out,
    total_rows,
    local_len: tl.constexpr,
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

    # Tile layout: rows index [dst_rank, local_s, shard_head], cols index hidden.
    # The q/k/v dimension is handled inside this one program so the pre pass still
    # emits the exact rank-major [W, S_local, 3, H/W, D] all-to-all layout.
    rows_per_rank: tl.constexpr = local_len * shard_heads
    rank = rows // rows_per_rank
    row_in_rank = rows - rank * rows_per_rank
    h = row_in_rank % shard_heads
    s = row_in_rank // shard_heads

    src_head = head_offset + rank * head_stride + h
    src_offsets = (s[:, None] * full_heads + src_head[:, None]) * hidden_dims + cols[None, :]

    q_row = s * 3 * shard_heads + h
    out_base = rank[:, None] * (local_len * 3 * shard_heads * hidden_dims) + q_row[:, None] * hidden_dims + cols[None, :]
    tl.store(out + out_base, tl.load(q + src_offsets, mask=mask), mask=mask)
    tl.store(out + out_base + shard_heads * hidden_dims, tl.load(k + src_offsets, mask=mask), mask=mask)
    tl.store(out + out_base + 2 * shard_heads * hidden_dims, tl.load(v + src_offsets, mask=mask), mask=mask)


@triton.jit
def _qkv_post_kernel(
    qkv,
    q_source,
    k_source,
    v_source,
    out_q,
    out_k,
    out_v,
    total_rows,
    aux_len,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    full_heads: tl.constexpr,
    head_offset: tl.constexpr,
    head_stride: tl.constexpr,
    main_first: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_d)
    row_mask = rows < total_rows
    col_mask = cols < hidden_dims
    mask = row_mask[:, None] & col_mask[None, :]

    # Tile layout: rows index [global_or_aux_s, local_shard_head], cols index hidden.
    # This mirrors the FP8 post kernel without dequantization: one program now handles
    # several complete hidden rows, so each row's image/text routing is computed once.
    h = rows % shard_heads
    out_s = rows // shard_heads

    final_len = total_rows // shard_heads
    global_main_len = final_len - aux_len
    local_main_len = global_main_len // world_size
    if main_first:
        is_main = out_s < global_main_len
        main_global_s = out_s
        aux_s = out_s - global_main_len
    else:
        is_main = out_s >= aux_len
        main_global_s = out_s - aux_len
        aux_s = out_s

    main_rank = main_global_s // local_main_len
    main_s = main_global_s - main_rank * local_main_len
    qkv_base = ((main_rank[:, None] * local_main_len + main_s[:, None]) * 3 * shard_heads + h[:, None]) * hidden_dims + cols[None, :]

    source_head = head_offset + cur_rank * head_stride + h
    aux_base = (aux_s[:, None] * full_heads + source_head[:, None]) * hidden_dims + cols[None, :]

    main_mask = mask & is_main[:, None]
    q_main = tl.load(qkv + qkv_base, mask=main_mask, other=0.0)
    k_main = tl.load(qkv + qkv_base + shard_heads * hidden_dims, mask=main_mask, other=0.0)
    v_main = tl.load(qkv + qkv_base + 2 * shard_heads * hidden_dims, mask=main_mask, other=0.0)

    aux_mask = mask & ~is_main[:, None]
    q_aux = tl.load(q_source + aux_base, mask=aux_mask, other=0.0)
    k_aux = tl.load(k_source + aux_base, mask=aux_mask, other=0.0)
    v_aux = tl.load(v_source + aux_base, mask=aux_mask, other=0.0)

    out_offsets = rows[:, None] * hidden_dims + cols[None, :]
    tl.store(out_q + out_offsets, q_main + q_aux, mask=mask)
    tl.store(out_k + out_offsets, k_main + k_aux, mask=mask)
    tl.store(out_v + out_offsets, v_main + v_aux, mask=mask)


@triton.jit
def _qonly_qkv_post_kernel(
    qkv,
    k_source,
    v_source,
    out_q,
    out_k,
    out_v,
    total_rows,
    q_rows,
    aux_len,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    full_heads: tl.constexpr,
    head_offset: tl.constexpr,
    head_stride: tl.constexpr,
    main_first: tl.constexpr,
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

    global_main_len = q_rows // shard_heads
    local_main_len = global_main_len // world_size
    kv_final_len = global_main_len + aux_len
    kv_rows = kv_final_len * shard_heads

    is_q_row = rows < q_rows
    q_main_rank = out_s // local_main_len
    q_main_s = out_s - q_main_rank * local_main_len
    q_row = q_main_s * 3 * shard_heads + h
    q_base = ((q_main_rank[:, None] * local_main_len + q_main_s[:, None]) * 3 * shard_heads + h[:, None]) * hidden_dims + cols[None, :]
    q_val = tl.load(qkv + q_base, mask=mask & is_q_row[:, None], other=0.0)
    tl.store(out_q + rows[:, None] * hidden_dims + cols[None, :], q_val, mask=mask & is_q_row[:, None])

    is_kv_row = rows < kv_rows
    if main_first:
        is_main = out_s < global_main_len
        main_global_s = out_s
        aux_s = out_s - global_main_len
    else:
        is_main = out_s >= aux_len
        main_global_s = out_s - aux_len
        aux_s = out_s

    main_rank = main_global_s // local_main_len
    main_s = main_global_s - main_rank * local_main_len
    k_main_base = ((main_rank[:, None] * local_main_len + main_s[:, None]) * 3 * shard_heads + h[:, None] + shard_heads) * hidden_dims + cols[None, :]
    v_main_base = k_main_base + shard_heads * hidden_dims
    main_mask = mask & is_kv_row[:, None] & is_main[:, None]
    k_main = tl.load(qkv + k_main_base, mask=main_mask, other=0.0)
    v_main = tl.load(qkv + v_main_base, mask=main_mask, other=0.0)

    source_head = head_offset + cur_rank * head_stride + h
    aux_base = (aux_s[:, None] * full_heads + source_head[:, None]) * hidden_dims + cols[None, :]
    aux_mask = mask & is_kv_row[:, None] & ~is_main[:, None]
    k_aux = tl.load(k_source + aux_base, mask=aux_mask, other=0.0)
    v_aux = tl.load(v_source + aux_base, mask=aux_mask, other=0.0)

    kv_offsets = rows[:, None] * hidden_dims + cols[None, :]
    kv_mask = mask & is_kv_row[:, None]
    tl.store(out_k + kv_offsets, k_main + k_aux, mask=kv_mask)
    tl.store(out_v + kv_offsets, v_main + v_aux, mask=kv_mask)


@triton.jit
def _attn_pre_kernel(
    attn,
    out,
    total_rows,
    local_len: tl.constexpr,
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

    # Tile layout: rows index [dst_rank, shard_head, local_s], cols index hidden.
    # This keeps each output row contiguous while avoiding per-element rank/head math.
    s = rows % local_len
    tmp = rows // local_len
    h = tmp % shard_heads
    rank = tmp // shard_heads

    src_s = rank * local_len + s
    src_offsets = src_s[:, None] * (shard_heads * hidden_dims) + h[:, None] * hidden_dims + cols[None, :]
    out_offsets = rows[:, None] * hidden_dims + cols[None, :]
    tl.store(out + out_offsets, tl.load(attn + src_offsets, mask=mask), mask=mask)


@triton.jit
def _attn_post_kernel(
    attn,
    out,
    total_rows,
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

    # Tile layout: rows index [local_s, rank, shard_head], cols index hidden.
    hidden_rows_per_seq = world_size * shard_heads
    h = rows % shard_heads
    tmp = rows // shard_heads
    rank = tmp % world_size
    out_s = tmp // world_size

    local_len = total_rows // hidden_rows_per_seq
    input_offsets = ((rank[:, None] * shard_heads + h[:, None]) * local_len + out_s[:, None]) * hidden_dims + cols[None, :]
    out_offsets = rows[:, None] * hidden_dims + cols[None, :]
    tl.store(out + out_offsets, tl.load(attn + input_offsets, mask=mask), mask=mask)


# FP8 quantized layout kernels

@triton.jit
def _qkv_pre_fp8_kernel(
    q,
    k,
    v,
    payload_fp8,
    scale_ptr,
    local_len: tl.constexpr,
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
    # One program handles the q/k/v triplet for one [dst_rank, local_s, shard_head].
    # That keeps the output layout unchanged while cutting launch work from 3 rows to 1.
    row = tl.program_id(0)
    offs = tl.arange(0, block_d)
    mask = offs < hidden_dims

    rows_per_rank: tl.constexpr = local_len * shard_heads
    rank = row // rows_per_rank
    row_in_rank = row - rank * rows_per_rank
    h = row_in_rank % shard_heads
    s = row_in_rank // shard_heads

    src_head = head_offset + rank * head_stride + h
    src_offsets = (s * full_heads + src_head) * hidden_dims + offs

    q_vals = tl.load(q + src_offsets, mask=mask, other=0.0).to(tl.float32)
    k_vals = tl.load(k + src_offsets, mask=mask, other=0.0).to(tl.float32)
    v_vals = tl.load(v + src_offsets, mask=mask, other=0.0).to(tl.float32)

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
def _qkv_post_fp8_kernel(
    payload_fp8,
    scale_ptr,
    q_source,
    k_source,
    v_source,
    out_q,
    out_k,
    out_v,
    total_rows,
    aux_len,
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
    main_first: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_d)
    row_mask = rows < total_rows
    col_mask = cols < hidden_dims
    mask = row_mask[:, None] & col_mask[None, :]

    # Tile layout: rows index [global_or_aux_s, local_shard_head], cols index hidden.
    # Each main row reads one fp8 payload row and one fp32 scale; scale is broadcast
    # across hidden instead of being reloaded for every element.
    h = rows % shard_heads
    out_s = rows // shard_heads

    final_len = total_rows // shard_heads
    global_main_len = final_len - aux_len
    local_main_len = global_main_len // world_size
    if main_first:
        is_main = out_s < global_main_len
        main_global_s = out_s
        aux_s = out_s - global_main_len
    else:
        is_main = out_s >= aux_len
        main_global_s = out_s - aux_len
        aux_s = out_s

    main_rank = main_global_s // local_main_len
    main_s = main_global_s - main_rank * local_main_len
    q_row = main_s * 3 * shard_heads + h
    k_row = q_row + shard_heads
    v_row = q_row + 2 * shard_heads

    q_payload = main_rank[:, None] * payload_elems_per_rank + q_row[:, None] * hidden_dims + cols[None, :]
    k_payload = main_rank[:, None] * payload_elems_per_rank + k_row[:, None] * hidden_dims + cols[None, :]
    v_payload = main_rank[:, None] * payload_elems_per_rank + v_row[:, None] * hidden_dims + cols[None, :]
    q_scale_offset = main_rank * scale_elems_per_rank + scale_base_offset + q_row
    k_scale_offset = main_rank * scale_elems_per_rank + scale_base_offset + k_row
    v_scale_offset = main_rank * scale_elems_per_rank + scale_base_offset + v_row

    main_mask = mask & is_main[:, None]
    q_scale = tl.load(scale_ptr + q_scale_offset, mask=row_mask & is_main, other=0.0).to(tl.float32)[:, None]
    k_scale = tl.load(scale_ptr + k_scale_offset, mask=row_mask & is_main, other=0.0).to(tl.float32)[:, None]
    v_scale = tl.load(scale_ptr + v_scale_offset, mask=row_mask & is_main, other=0.0).to(tl.float32)[:, None]
    q_main = tl.load(payload_fp8 + q_payload, mask=main_mask, other=0.0).to(tl.float32) * q_scale
    k_main = tl.load(payload_fp8 + k_payload, mask=main_mask, other=0.0).to(tl.float32) * k_scale
    v_main = tl.load(payload_fp8 + v_payload, mask=main_mask, other=0.0).to(tl.float32) * v_scale

    source_head = head_offset + cur_rank * head_stride + h
    aux_base = (aux_s[:, None] * full_heads + source_head[:, None]) * hidden_dims + cols[None, :]
    aux_mask = mask & ~is_main[:, None]
    q_aux = tl.load(q_source + aux_base, mask=aux_mask, other=0.0).to(tl.float32)
    k_aux = tl.load(k_source + aux_base, mask=aux_mask, other=0.0).to(tl.float32)
    v_aux = tl.load(v_source + aux_base, mask=aux_mask, other=0.0).to(tl.float32)

    out_offsets = rows[:, None] * hidden_dims + cols[None, :]
    tl.store(out_q + out_offsets, q_main + q_aux, mask=mask)
    tl.store(out_k + out_offsets, k_main + k_aux, mask=mask)
    tl.store(out_v + out_offsets, v_main + v_aux, mask=mask)


@triton.jit
def _qonly_qkv_post_fp8_kernel(
    payload_fp8,
    scale_ptr,
    k_source,
    v_source,
    out_q,
    out_k,
    out_v,
    total_rows,
    q_rows,
    aux_len,
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
    main_first: tl.constexpr,
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

    global_main_len = q_rows // shard_heads
    local_main_len = global_main_len // world_size
    kv_final_len = global_main_len + aux_len
    kv_rows = kv_final_len * shard_heads

    is_q_row = rows < q_rows
    q_main_rank = out_s // local_main_len
    q_main_s = out_s - q_main_rank * local_main_len
    q_row = q_main_s * 3 * shard_heads + h
    q_payload = q_main_rank[:, None] * payload_elems_per_rank + q_row[:, None] * hidden_dims + cols[None, :]
    q_scale_offset = q_main_rank * scale_elems_per_rank + scale_base_offset + q_row
    q_scale = tl.load(scale_ptr + q_scale_offset, mask=row_mask & is_q_row, other=0.0).to(tl.float32)[:, None]
    q_val = tl.load(payload_fp8 + q_payload, mask=mask & is_q_row[:, None], other=0.0).to(tl.float32) * q_scale
    tl.store(out_q + rows[:, None] * hidden_dims + cols[None, :], q_val, mask=mask & is_q_row[:, None])

    is_kv_row = rows < kv_rows
    if main_first:
        is_main = out_s < global_main_len
        main_global_s = out_s
        aux_s = out_s - global_main_len
    else:
        is_main = out_s >= aux_len
        main_global_s = out_s - aux_len
        aux_s = out_s

    main_rank = main_global_s // local_main_len
    main_s = main_global_s - main_rank * local_main_len
    qkv_row = main_s * 3 * shard_heads + h
    k_row = qkv_row + shard_heads
    v_row = qkv_row + 2 * shard_heads
    k_payload = main_rank[:, None] * payload_elems_per_rank + k_row[:, None] * hidden_dims + cols[None, :]
    v_payload = main_rank[:, None] * payload_elems_per_rank + v_row[:, None] * hidden_dims + cols[None, :]
    k_scale_offset = main_rank * scale_elems_per_rank + scale_base_offset + k_row
    v_scale_offset = main_rank * scale_elems_per_rank + scale_base_offset + v_row
    main_mask = mask & is_kv_row[:, None] & is_main[:, None]
    k_scale = tl.load(scale_ptr + k_scale_offset, mask=row_mask & is_kv_row & is_main, other=0.0).to(tl.float32)[:, None]
    v_scale = tl.load(scale_ptr + v_scale_offset, mask=row_mask & is_kv_row & is_main, other=0.0).to(tl.float32)[:, None]
    k_main = tl.load(payload_fp8 + k_payload, mask=main_mask, other=0.0).to(tl.float32) * k_scale
    v_main = tl.load(payload_fp8 + v_payload, mask=main_mask, other=0.0).to(tl.float32) * v_scale

    source_head = head_offset + cur_rank * head_stride + h
    aux_base = (aux_s[:, None] * full_heads + source_head[:, None]) * hidden_dims + cols[None, :]
    aux_mask = mask & is_kv_row[:, None] & ~is_main[:, None]
    k_aux = tl.load(k_source + aux_base, mask=aux_mask, other=0.0).to(tl.float32)
    v_aux = tl.load(v_source + aux_base, mask=aux_mask, other=0.0).to(tl.float32)

    kv_offsets = rows[:, None] * hidden_dims + cols[None, :]
    kv_mask = mask & is_kv_row[:, None]
    tl.store(out_k + kv_offsets, k_main + k_aux, mask=kv_mask)
    tl.store(out_v + kv_offsets, v_main + v_aux, mask=kv_mask)


@triton.jit
def _attn_pre_fp8_kernel(
    attn,
    payload_fp8,
    scale_ptr,
    local_len: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    payload_elems_per_rank: tl.constexpr,
    scale_base_offset: tl.constexpr,
    scale_elems_per_rank: tl.constexpr,
    block_d: tl.constexpr,
):
    # One program handles one [shard_head, local_s] row and writes payload plus scale.
    row = tl.program_id(0)
    offs = tl.arange(0, block_d)
    mask = offs < hidden_dims

    rows_per_rank: tl.constexpr = shard_heads * local_len
    rank = row // rows_per_rank
    row_in_rank = row - rank * rows_per_rank
    s = row_in_rank % local_len
    h = row_in_rank // local_len

    src_s = rank * local_len + s
    src_offsets = src_s * (shard_heads * hidden_dims) + h * hidden_dims + offs
    vals = tl.load(attn + src_offsets, mask=mask, other=0.0).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(vals), axis=0), 0.001953125)
    scale = amax / 448.0
    quant = (vals / scale).to(tl.float8e4nv)

    payload_offsets = rank * payload_elems_per_rank + row_in_rank * hidden_dims + offs
    tl.store(payload_fp8 + payload_offsets, quant, mask=mask)
    scale_offset = rank * scale_elems_per_rank + scale_base_offset + row_in_rank
    tl.store(scale_ptr + scale_offset, scale)


@triton.jit
def _attn_post_fp8_kernel(
    payload_fp8,
    scale_ptr,
    out,
    total_rows,
    world_size: tl.constexpr,
    shard_heads: tl.constexpr,
    hidden_dims: tl.constexpr,
    payload_elems_per_rank: tl.constexpr,
    scale_base_offset: tl.constexpr,
    scale_elems_per_rank: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_d)
    row_mask = rows < total_rows
    col_mask = cols < hidden_dims
    mask = row_mask[:, None] & col_mask[None, :]

    # Tile layout: rows index [local_s, rank, shard_head], cols index hidden.
    hidden_rows_per_seq = world_size * shard_heads
    h = rows % shard_heads
    tmp = rows // shard_heads
    rank = tmp % world_size
    out_s = tmp // world_size

    local_len = total_rows // hidden_rows_per_seq
    row_in_rank = h * local_len + out_s
    payload_offset = rank[:, None] * payload_elems_per_rank + row_in_rank[:, None] * hidden_dims + cols[None, :]
    scale_offset = rank * scale_elems_per_rank + scale_base_offset + row_in_rank
    scale = tl.load(scale_ptr + scale_offset, mask=row_mask, other=0.0).to(tl.float32)[:, None]
    value = tl.load(payload_fp8 + payload_offset, mask=mask, other=0.0).to(tl.float32) * scale

    out_offsets = rows[:, None] * hidden_dims + cols[None, :]
    tl.store(out + out_offsets, value, mask=mask)


# Public BF16/FP16 wrappers

def qkv_pre(q, k, v, world_size, block_size=None, head_index=None):
    """Prepare BF16/FP16 q/k/v for the first Ulysses all-to-all.

    Inputs are this rank's main-token tensors ``[S_local, H, D]``.
    The output is rank-major ``[W, S_local, 3, H_out, D]`` so each
    all-to-all split is already contiguous for one destination rank. ``H_out``
    is ``H/W`` for bulk execution and one when ``head_index`` selects a lane.
    """
    _check_cuda_contiguous(q, k, v)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v must have the same shape for qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    local_len, heads, hidden_dims = q.shape
    _check_hidden_dims(hidden_dims, "fused QKV pre path")

    shard_heads, head_offset, head_stride = _resolve_qkv_head_layout(heads, world_size, head_index)
    out = torch.empty((world_size, local_len, 3, shard_heads, hidden_dims), device=q.device, dtype=q.dtype)
    total_rows = world_size * local_len * shard_heads
    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _qkv_pre_kernel[grid](
        q,
        k,
        v,
        out,
        total_rows,
        local_len,
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


def qkv_post(qkv, q_source, k_source, v_source, cur_rank, aux_len, main_first, q_only=False, block_size=None, head_index=None):
    """Finish the first all-to-all and optionally join an auxiliary token region.

    ``q_only=True`` keeps q main-only while joining the auxiliary region to k/v.
    Separate Triton kernels preserve the simpler hot path for each output contract.
    """
    _check_cuda_contiguous(qkv, q_source, k_source, v_source)
    world_size, main_local_len, qkv_count, shard_heads, hidden_dims = qkv.shape
    if qkv_count != 3:
        raise ValueError(f"qkv fused buffer must have qkv dimension == 3, got qkv_count={qkv_count}.")
    if q_only:
        if k_source.shape != v_source.shape:
            raise ValueError(f"k/v source tensors must have the same shape, got k={tuple(k_source.shape)}, v={tuple(v_source.shape)}.")
        full_heads = k_source.shape[1]
    else:
        if q_source.shape != k_source.shape or q_source.shape != v_source.shape:
            raise ValueError(
                f"q/k/v source tensors must have the same shape, got q={tuple(q_source.shape)}, "
                f"k={tuple(k_source.shape)}, v={tuple(v_source.shape)}."
            )
        full_heads = q_source.shape[1]
    expected_shard_heads, head_offset, head_stride = _resolve_qkv_head_layout(full_heads, world_size, head_index)
    if shard_heads != expected_shard_heads:
        raise ValueError(f"qkv shard_heads must be {expected_shard_heads}, got {shard_heads}.")
    _check_hidden_dims(hidden_dims, "fused QKV post path")

    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    main_global_len = world_size * main_local_len

    if q_only:
        q_shape = (main_global_len, shard_heads, hidden_dims)
        kv_shape = (main_global_len + aux_len, shard_heads, hidden_dims)
        out_q = torch.empty(q_shape, device=qkv.device, dtype=qkv.dtype)
        out_k = torch.empty(kv_shape, device=qkv.device, dtype=qkv.dtype)
        out_v = torch.empty_like(out_k)
        q_rows = q_shape[0] * shard_heads
        total_rows = max(q_rows, kv_shape[0] * shard_heads)
        grid = (triton.cdiv(total_rows, block_m),)
        _qonly_qkv_post_kernel[grid](
            qkv,
            k_source,
            v_source,
            out_q,
            out_k,
            out_v,
            total_rows,
            q_rows,
            aux_len,
            cur_rank,
            world_size,
            shard_heads,
            hidden_dims,
            full_heads,
            head_offset,
            head_stride,
            main_first,
            block_m,
            block_d,
            num_warps=4,
        )
        return out_q, out_k, out_v

    shape = (main_global_len + aux_len, shard_heads, hidden_dims)
    out_q = torch.empty(shape, device=qkv.device, dtype=qkv.dtype)
    out_k = torch.empty_like(out_q)
    out_v = torch.empty_like(out_q)
    total_rows = shape[0] * shard_heads
    grid = (triton.cdiv(total_rows, block_m),)
    _qkv_post_kernel[grid](
        qkv,
        q_source,
        k_source,
        v_source,
        out_q,
        out_k,
        out_v,
        total_rows,
        aux_len,
        cur_rank,
        world_size,
        shard_heads,
        hidden_dims,
        full_heads,
        head_offset,
        head_stride,
        main_first,
        block_m,
        block_d,
        num_warps=4,
    )
    return out_q, out_k, out_v


def attn_pre(attn, local_len, world_size, shard_heads, hidden_dims, block_size=None):
    """Prepare BF16/FP16 attention output for the second all-to-all.

    ``attn`` is the local-head attention result ``[W * S_local, H/W * D]``.
    Rows are grouped by source rank and emitted as ``[W, H/W, S_local, D]``.
    """
    _check_cuda_contiguous(attn)
    _check_hidden_dims(hidden_dims, "fused attn pre path")

    out = torch.empty((world_size, shard_heads, local_len, hidden_dims), device=attn.device, dtype=attn.dtype)
    total_rows = world_size * shard_heads * local_len
    block_m = _resolve_block_m(block_size, hidden_dims, default=16)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _attn_pre_kernel[grid](
        attn,
        out,
        total_rows,
        local_len,
        shard_heads,
        hidden_dims,
        block_m,
        block_d,
        num_warps=4,
    )
    return out


def attn_post(attn, block_size=None):
    """Finish the second all-to-all and return ``[S_local, H * D]``."""
    _check_cuda_contiguous(attn)
    world_size, shard_heads, local_len, hidden_dims = attn.shape
    _check_hidden_dims(hidden_dims, "fused attn post path")

    out = torch.empty((local_len, world_size * shard_heads * hidden_dims), device=attn.device, dtype=attn.dtype)
    total_rows = local_len * world_size * shard_heads
    block_m = _resolve_block_m(block_size, hidden_dims, default=8)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _attn_post_kernel[grid](
        attn,
        out,
        total_rows,
        world_size,
        shard_heads,
        hidden_dims,
        block_m,
        block_d,
        num_warps=4,
    )
    return out


# Public FP8 wrappers

def qkv_pre_fp8(q, k, v, world_size, head_index=None):
    """FP8 version of ``qkv_pre`` with fused layout+quantization.

    Returns separate payload/scale tensors so runtime communication can keep
    the faster legacy NCCL shape: one all-to-all for FP8 payload and one for
    FP32 per-row scale.
    """
    _check_cuda_contiguous(q, k, v)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v must have the same shape for qkv fusion, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")
    local_len, heads, hidden_dims = q.shape
    _check_hidden_dims(hidden_dims, "fused QKV fp8 pre path")

    shard_heads, head_offset, head_stride = _resolve_qkv_head_layout(heads, world_size, head_index)
    payload_shape = (world_size, local_len, 3, shard_heads, hidden_dims)
    scale_shape = (*payload_shape[:-1], 1)
    payload_elems_per_rank, scale_elems_per_rank = _fp8_split_layout(payload_shape, scale_shape, name="fused QKV fp8 split path")

    payload = torch.empty(payload_shape, device=q.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(scale_shape, device=q.device, dtype=torch.float32)
    total_rows = world_size * local_len * shard_heads
    block_d = _next_power_of_2(hidden_dims)
    _qkv_pre_fp8_kernel[(total_rows,)](
        q,
        k,
        v,
        payload,
        scale,
        local_len,
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


def qkv_post_fp8(
    payload,
    scale,
    payload_shape,
    scale_shape,
    q_source,
    k_source,
    v_source,
    cur_rank,
    aux_len,
    main_first,
    q_only=False,
    block_size=None,
    head_index=None,
):
    """FP8 version of ``qkv_post`` for separate payload/scale tensors."""
    _check_cuda_contiguous(payload, scale, q_source, k_source, v_source)
    world_size, main_local_len, qkv_count, shard_heads, hidden_dims = payload_shape
    if qkv_count != 3:
        raise ValueError(f"qkv fused buffer must have qkv dimension == 3, got qkv_count={qkv_count}.")
    if tuple(scale_shape) != (world_size, main_local_len, 3, shard_heads, 1):
        raise ValueError("qkv scale shape does not match payload shape.")
    if q_only:
        if k_source.shape != v_source.shape:
            raise ValueError(f"k/v source tensors must have the same shape, got k={tuple(k_source.shape)}, v={tuple(v_source.shape)}.")
        full_heads = k_source.shape[1]
    else:
        if q_source.shape != k_source.shape or q_source.shape != v_source.shape:
            raise ValueError(
                f"q/k/v source tensors must have the same shape, got q={tuple(q_source.shape)}, "
                f"k={tuple(k_source.shape)}, v={tuple(v_source.shape)}."
            )
        full_heads = q_source.shape[1]
    expected_shard_heads, head_offset, head_stride = _resolve_qkv_head_layout(full_heads, world_size, head_index)
    if shard_heads != expected_shard_heads:
        raise ValueError(f"qkv shard_heads must be {expected_shard_heads}, got {shard_heads}.")

    _check_hidden_dims(hidden_dims, "fused QKV fp8 post path")
    payload_elems_per_rank, scale_elems_per_rank = _check_fp8_split(payload, scale, payload_shape, scale_shape, torch.float32, name="split QKV fp8 input")

    block_m = _resolve_block_m(block_size, hidden_dims, default=16)
    block_d = _next_power_of_2(hidden_dims)
    main_global_len = world_size * main_local_len

    if q_only:
        q_shape = (main_global_len, shard_heads, hidden_dims)
        kv_shape = (main_global_len + aux_len, shard_heads, hidden_dims)
        out_q = torch.empty(q_shape, device=payload.device, dtype=k_source.dtype)
        out_k = torch.empty(kv_shape, device=payload.device, dtype=k_source.dtype)
        out_v = torch.empty_like(out_k)
        q_rows = q_shape[0] * shard_heads
        total_rows = max(q_rows, kv_shape[0] * shard_heads)
        grid = (triton.cdiv(total_rows, block_m),)
        _qonly_qkv_post_fp8_kernel[grid](
            payload,
            scale,
            k_source,
            v_source,
            out_q,
            out_k,
            out_v,
            total_rows,
            q_rows,
            aux_len,
            cur_rank,
            world_size,
            shard_heads,
            hidden_dims,
            full_heads,
            head_offset,
            head_stride,
            payload_elems_per_rank,
            0,
            scale_elems_per_rank,
            main_first,
            block_m,
            block_d,
            num_warps=4,
        )
        return out_q, out_k, out_v

    shape = (main_global_len + aux_len, shard_heads, hidden_dims)
    out_q = torch.empty(shape, device=payload.device, dtype=q_source.dtype)
    out_k = torch.empty_like(out_q)
    out_v = torch.empty_like(out_q)
    total_rows = shape[0] * shard_heads
    grid = (triton.cdiv(total_rows, block_m),)
    _qkv_post_fp8_kernel[grid](
        payload,
        scale,
        q_source,
        k_source,
        v_source,
        out_q,
        out_k,
        out_v,
        total_rows,
        aux_len,
        cur_rank,
        world_size,
        shard_heads,
        hidden_dims,
        full_heads,
        head_offset,
        head_stride,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        main_first,
        block_m,
        block_d,
        num_warps=4,
    )
    return out_q, out_k, out_v


def attn_pre_fp8(attn, local_len, world_size, shard_heads, hidden_dims):
    """FP8 version of ``attn_pre`` with fused layout+quantization."""
    _check_cuda_contiguous(attn)
    _check_hidden_dims(hidden_dims, "fused attn fp8 pre path")

    payload_shape = (world_size, shard_heads, local_len, hidden_dims)
    scale_shape = (*payload_shape[:-1], 1)
    payload_elems_per_rank, scale_elems_per_rank = _fp8_split_layout(payload_shape, scale_shape, name="fused attn fp8 split path")

    payload = torch.empty(payload_shape, device=attn.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(scale_shape, device=attn.device, dtype=torch.float32)
    total_rows = world_size * shard_heads * local_len
    block_d = _next_power_of_2(hidden_dims)
    _attn_pre_fp8_kernel[(total_rows,)](
        attn,
        payload,
        scale,
        local_len,
        shard_heads,
        hidden_dims,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        block_d,
        num_warps=4,
    )
    return payload, scale, payload_shape, scale_shape


def attn_post_fp8(payload, scale, payload_shape, scale_shape, output_dtype, block_size=None):
    """FP8 version of ``attn_post`` for separate payload/scale tensors."""
    _check_cuda_contiguous(payload, scale)
    world_size, shard_heads, local_len, hidden_dims = payload_shape
    if tuple(scale_shape) != (world_size, shard_heads, local_len, 1):
        raise ValueError("attn scale shape does not match payload shape.")

    _check_hidden_dims(hidden_dims, "fused attn fp8 post path")
    payload_elems_per_rank, scale_elems_per_rank = _check_fp8_split(payload, scale, payload_shape, scale_shape, torch.float32, name="split attn fp8 input")

    out = torch.empty((local_len, world_size * shard_heads * hidden_dims), device=payload.device, dtype=output_dtype)
    total_rows = local_len * world_size * shard_heads
    block_m = _resolve_block_m(block_size, hidden_dims, default=16)
    block_d = _next_power_of_2(hidden_dims)
    grid = (triton.cdiv(total_rows, block_m),)
    _attn_post_fp8_kernel[grid](
        payload,
        scale,
        out,
        total_rows,
        world_size,
        shard_heads,
        hidden_dims,
        payload_elems_per_rank,
        0,
        scale_elems_per_rank,
        block_m,
        block_d,
        num_warps=4,
    )
    return out
