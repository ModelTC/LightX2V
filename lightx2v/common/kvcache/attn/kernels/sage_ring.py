"""Incremental SageAttention2 quantization and LSE reduction for a physical KV ring."""

import triton
import triton.language as tl


@triton.jit
def segment_is_dirty(metadata, valid, part, write_tokens: tl.constexpr, capacity: tl.constexpr, segment_tokens: tl.constexpr):
    start = tl.load(metadata)
    sink = tl.load(metadata + 1)
    head = tl.load(metadata + 2)
    end = start + write_tokens
    part_start = part * segment_tokens
    part_end = tl.minimum(part_start + segment_tokens, capacity)
    sink_overlap = (part_start < tl.minimum(end, sink)) & (part_end > start)
    recent_start = tl.maximum(start, sink)
    recent_tokens = tl.maximum(end - recent_start, 0)
    physical_start = sink + (head + recent_start - sink) % (capacity - sink)
    physical_end = physical_start + recent_tokens
    first_overlap = (part_start < tl.minimum(physical_end, capacity)) & (part_end > physical_start)
    second_overlap = (part_start < sink + tl.maximum(physical_end - capacity, 0)) & (part_end > sink)
    return (tl.load(valid + part) == 0) | sink_overlap | (recent_tokens > 0) & (first_overlap | second_overlap)


@triton.jit
def segment_stats(
    key,
    value,
    metadata,
    valid,
    key_mean,
    value_scale,
    write_tokens: tl.constexpr,
    capacity: tl.constexpr,
    visible_tokens: tl.constexpr,
    segment_tokens: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    scale_max: tl.constexpr,
    columns_per_block: tl.constexpr,
):
    part = tl.program_id(0)
    head = tl.program_id(1)
    if segment_is_dirty(metadata, valid, part, write_tokens, capacity, segment_tokens):
        columns = tl.program_id(2) * columns_per_block + tl.arange(0, columns_per_block)
        rows = tl.arange(0, 64)
        length = tl.minimum(segment_tokens, visible_tokens - part * segment_tokens)
        key_sum = tl.full((columns_per_block,), 0, tl.float32)
        value_max = tl.full((columns_per_block,), 0, tl.float32)
        for offset in range(0, length, 64):
            local_rows = offset + rows
            addresses = ((part * segment_tokens + local_rows[:, None]) * heads + head) * dim + columns[None, :]
            mask = (local_rows[:, None] < length) & (columns[None, :] < dim)
            keys = tl.load(key + addresses, mask, 0).to(tl.float32)
            values = tl.load(value + addresses, mask, 0).to(tl.float32)
            key_sum += tl.sum(keys, 0)
            value_max = tl.maximum(value_max, tl.max(tl.abs(values), 0))
        address = (part * heads + head) * dim + columns
        tl.store(key_mean + address, key_sum / length, columns < dim)
        tl.store(value_scale + address, tl.maximum(value_max, 1.0e-12) / scale_max, columns < dim)


@triton.jit
def quantize_key(
    key,
    metadata,
    valid,
    key_mean,
    key_int8,
    key_scale,
    write_tokens: tl.constexpr,
    capacity: tl.constexpr,
    visible_tokens: tl.constexpr,
    segment_tokens: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    key_block: tl.constexpr,
    per_thread: tl.constexpr,
    scale_tokens: tl.constexpr,
):
    part = tl.program_id(0)
    head = tl.program_id(1)
    group = tl.program_id(2)
    if segment_is_dirty(metadata, valid, part, write_tokens, capacity, segment_tokens):
        columns = tl.arange(0, dim)
        if per_thread:
            block = group // 4
            lane = group % 4
            rows0 = block * key_block + tl.arange(0, key_block // 8) * 8 + lane * 2
            rows1 = rows0 + 1
            length = tl.minimum(segment_tokens, visible_tokens - part * segment_tokens)
            center = tl.load(key_mean + (part * heads + head) * dim + columns)
            address0 = ((part * segment_tokens + rows0[:, None]) * heads + head) * dim + columns[None, :]
            address1 = ((part * segment_tokens + rows1[:, None]) * heads + head) * dim + columns[None, :]
            x0 = tl.load(key + address0, rows0[:, None] < length, 0)
            x1 = tl.load(key + address1, rows1[:, None] < length, 0)
            x0 = (x0.to(tl.float32) - center[None, :].to(tl.float32)).to(key.dtype.element_ty).to(tl.float32)
            x1 = (x1.to(tl.float32) - center[None, :].to(tl.float32)).to(key.dtype.element_ty).to(tl.float32)
            x0 = tl.where(rows0[:, None] < length, x0, 0.0)
            x1 = tl.where(rows1[:, None] < length, x1, 0.0)
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
            length = tl.minimum(segment_tokens, visible_tokens - part * segment_tokens)
            center = tl.load(key_mean + (part * heads + head) * dim + columns)
            address = ((part * segment_tokens + rows[:, None]) * heads + head) * dim + columns[None, :]
            x = tl.load(key + address, rows[:, None] < length, 0).to(tl.float32) - center[None, :].to(tl.float32)
            x = tl.where(rows[:, None] < length, x, 0.0)
            maximum = tl.maximum(tl.max(tl.abs(x)), 1.0e-7)
            factor = maximum / 127.0
            quantized = tl.extra.cuda.libdevice.nearbyint(x * (127.0 / maximum)).to(tl.int8)
            tl.store(key_int8 + address, quantized, rows[:, None] < segment_tokens)
        tl.store(key_scale + (part * heads + head) * scale_tokens + group, factor)


@triton.jit
def quantize_value(
    value,
    metadata,
    valid,
    value_scale,
    value_fp8,
    write_tokens: tl.constexpr,
    capacity: tl.constexpr,
    visible_tokens: tl.constexpr,
    segment_tokens: tl.constexpr,
    padded_tokens: tl.constexpr,
    scale_max: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
):
    part = tl.program_id(0)
    head = tl.program_id(1)
    if segment_is_dirty(metadata, valid, part, write_tokens, capacity, segment_tokens):
        rows = tl.program_id(2) // triton.cdiv(dim, 32) * 32 + tl.arange(0, 32)
        columns = tl.program_id(2) % triton.cdiv(dim, 32) * 32 + tl.arange(0, 32)
        length = tl.minimum(segment_tokens, visible_tokens - part * segment_tokens)
        address = ((part * segment_tokens + rows[:, None]) * heads + head) * dim + columns[None, :]
        x = tl.load(value + address, (rows[:, None] < length) & (columns[None, :] < dim), 0).to(tl.float32)
        scale = tl.load(value_scale + (part * heads + head) * dim + columns, columns < dim, 1.0)
        maximum = tl.maximum((scale * scale_max).to(value.dtype.element_ty).to(tl.float32), 1.0e-12)
        inverse_scale = scale_max / maximum
        quantized = x * inverse_scale[None, :]
        permuted = (rows & ~15) | (rows & 1) | ((rows & 8) >> 2) | ((rows & 2) << 1) | ((rows & 4) << 1)
        output = ((part * dim + columns[None, :]) * heads + head) * padded_tokens + permuted[:, None]
        tl.store(value_fp8 + output, quantized, (rows[:, None] < padded_tokens) & (columns[None, :] < dim))


@triton.jit
def mark_valid(valid, parts: tl.constexpr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    tl.store(valid + offsets, 1, offsets < parts)


@triton.jit
def merge_attention(
    partial_output,
    full_lse,
    tail_lse,
    query,
    key_mean,
    output,
    queries: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    full_parts: tl.constexpr,
    has_tail: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    columns = tl.arange(0, dim)
    q = tl.load(query + (row * heads + head) * dim + columns).to(tl.float32)
    top = -float("inf")
    denominator = 0.0
    accumulator = tl.full((dim,), 0, tl.float32)
    for part in tl.static_range(full_parts + has_tail):
        mean = tl.load(key_mean + (part * heads + head) * dim + columns).to(tl.float32)
        if part < full_parts:
            lse = tl.load(full_lse + (part * heads + head) * queries + row)
        else:
            lse = tl.load(tail_lse + head * queries + row)
        score = lse * 0.6931471805599453 + tl.sum(q * mean, 0) * dim**-0.5
        new_top = tl.maximum(top, score)
        old_weight = tl.exp(top - new_top)
        weight = tl.exp(score - new_top)
        partial = tl.load(partial_output + ((part * queries + row) * heads + head) * dim + columns).to(tl.float32)
        accumulator = accumulator * old_weight + partial * weight
        denominator = denominator * old_weight + weight
        top = new_top
    tl.store(output + (row * heads + head) * dim + columns, accumulator / denominator)
