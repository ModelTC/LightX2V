"""Encode complete current K/V segments without any raw historical storage."""

import triton
import triton.language as tl


@triton.jit
def destination_part(metadata, current_part, capacity: tl.constexpr, segment_tokens: tl.constexpr, physical_start: tl.constexpr):
    if physical_start >= 0:
        return physical_start // segment_tokens + current_part
    logical = tl.load(metadata) + current_part * segment_tokens
    sink, head = tl.load(metadata + 1), tl.load(metadata + 2)
    physical = tl.where(logical < sink, logical, sink + (head + logical - sink) % (capacity - sink))
    return physical // segment_tokens


@triton.jit
def current_stats(
    key,
    value,
    metadata,
    position,
    key_mean,
    value_scale,
    capacity: tl.constexpr,
    segment_tokens: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    key_stride_token: tl.constexpr,
    key_stride_head: tl.constexpr,
    key_stride_dim: tl.constexpr,
    value_stride_token: tl.constexpr,
    value_stride_head: tl.constexpr,
    value_stride_dim: tl.constexpr,
    scale_max: tl.constexpr,
    cache_fp8_values: tl.constexpr,
    physical_start: tl.constexpr,
    enabled_after: tl.constexpr,
):
    if enabled_after < 0 or tl.load(position) > enabled_after:
        current_part, head = tl.program_id(0), tl.program_id(1)
        part = destination_part(metadata, current_part, capacity, segment_tokens, physical_start)
        columns = tl.program_id(2) * 32 + tl.arange(0, 32)
        rows = tl.arange(0, 64)
        key_sum = tl.full((32,), 0, tl.float32)
        value_max = tl.full((32,), 0, tl.float32)
        for offset in range(0, segment_tokens, 64):
            local_rows = offset + rows
            current_rows = current_part * segment_tokens + local_rows
            mask = (local_rows[:, None] < segment_tokens) & (columns[None, :] < dim)
            addresses = current_rows[:, None] * key_stride_token + head * key_stride_head + columns[None, :] * key_stride_dim
            keys = tl.load(key + addresses, mask, 0).to(tl.float32)
            key_sum += tl.sum(keys, 0)
            if cache_fp8_values:
                addresses = current_rows[:, None] * value_stride_token + head * value_stride_head + columns[None, :] * value_stride_dim
                values = tl.load(value + addresses, mask, 0).to(tl.float32)
                value_max = tl.maximum(value_max, tl.max(tl.abs(values), 0))
        address = (part * heads + head) * dim + columns
        tl.store(key_mean + address, key_sum / segment_tokens, columns < dim)
        if cache_fp8_values:
            tl.store(value_scale + address, tl.maximum(value_max, 1.0e-12) / scale_max, columns < dim)


@triton.jit
def quantize_current_key(
    key,
    metadata,
    position,
    key_mean,
    key_int8,
    key_scale,
    key_valid,
    capacity: tl.constexpr,
    segment_tokens: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    key_stride_token: tl.constexpr,
    key_stride_head: tl.constexpr,
    key_stride_dim: tl.constexpr,
    key_block: tl.constexpr,
    per_thread: tl.constexpr,
    scale_tokens: tl.constexpr,
    physical_start: tl.constexpr,
    enabled_after: tl.constexpr,
):
    if enabled_after < 0 or tl.load(position) > enabled_after:
        current_part, head, group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        part = destination_part(metadata, current_part, capacity, segment_tokens, physical_start)
        columns = tl.arange(0, dim)
        center = tl.load(key_mean + (part * heads + head) * dim + columns)
        if per_thread:
            block, lane = group // 4, group % 4
            rows0 = block * key_block + tl.arange(0, key_block // 8) * 8 + lane * 2
            rows1 = rows0 + 1
            current_rows0 = current_part * segment_tokens + rows0
            current_rows1 = current_part * segment_tokens + rows1
            address0 = current_rows0[:, None] * key_stride_token + head * key_stride_head + columns[None, :] * key_stride_dim
            address1 = current_rows1[:, None] * key_stride_token + head * key_stride_head + columns[None, :] * key_stride_dim
            x0 = tl.load(key + address0, rows0[:, None] < segment_tokens, 0).to(tl.float32)
            x1 = tl.load(key + address1, rows1[:, None] < segment_tokens, 0).to(tl.float32)
            x0 = (x0 - center[None, :].to(tl.float32)).to(key.dtype.element_ty).to(tl.float32)
            x1 = (x1 - center[None, :].to(tl.float32)).to(key.dtype.element_ty).to(tl.float32)
            x0 = tl.where(rows0[:, None] < segment_tokens, x0, 0.0)
            x1 = tl.where(rows1[:, None] < segment_tokens, x1, 0.0)
            factor = tl.maximum(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 127.0 + 1.0e-7
            q0, q1 = x0 / factor, x1 / factor
            q0 = (q0 + 0.5 * tl.where(q0 >= 0, 1, -1)).to(tl.int8)
            q1 = (q1 + 0.5 * tl.where(q1 >= 0, 1, -1)).to(tl.int8)
            out0 = ((part * segment_tokens + rows0[:, None]) * heads + head) * dim + columns[None, :]
            out1 = ((part * segment_tokens + rows1[:, None]) * heads + head) * dim + columns[None, :]
            tl.store(key_int8 + out0, q0, rows0[:, None] < segment_tokens)
            tl.store(key_int8 + out1, q1, rows1[:, None] < segment_tokens)
        else:
            rows = group * key_block + tl.arange(0, key_block)
            current_rows = current_part * segment_tokens + rows
            addresses = current_rows[:, None] * key_stride_token + head * key_stride_head + columns[None, :] * key_stride_dim
            values = tl.load(key + addresses, rows[:, None] < segment_tokens, 0).to(tl.float32) - center[None, :].to(tl.float32)
            values = tl.where(rows[:, None] < segment_tokens, values, 0.0)
            maximum = tl.maximum(tl.max(tl.abs(values)), 1.0e-7)
            factor = maximum / 127.0
            quantized = tl.extra.cuda.libdevice.nearbyint(values * (127.0 / maximum)).to(tl.int8)
            output = ((part * segment_tokens + rows[:, None]) * heads + head) * dim + columns[None, :]
            tl.store(key_int8 + output, quantized, rows[:, None] < segment_tokens)
        tl.store(key_scale + (part * heads + head) * scale_tokens + group, factor)
        if head == 0 and group == 0:
            tl.store(key_valid + part, 1)
