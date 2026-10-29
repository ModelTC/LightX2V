"""Incremental SageAttention2 quantization and LSE reduction for a physical KV ring."""

import triton
import triton.language as tl


@triton.jit
def rotate_query(query, output, freqs, metadata, count: tl.constexpr, heads: tl.constexpr, dim: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < count
    column = offsets % dim
    token = offsets // (heads * dim)
    logical = tl.load(metadata) + token
    value = tl.load(query + offsets, mask, 0).to(tl.float32)
    partner = tl.load(query + (offsets ^ 1), mask, 0).to(tl.float32)
    cosine = tl.load(freqs + logical * dim + column // 2, mask, 0)
    sine = tl.load(freqs + logical * dim + dim // 2 + column // 2, mask, 0)
    product = partner * sine
    result = tl.where(column % 2 == 0, tl.fma(value, cosine, -product), tl.fma(value, cosine, product))
    tl.store(output + offsets, result, mask)


@triton.jit
def physical_rows(rows, metadata, capacity: tl.constexpr, logical_start: tl.constexpr):
    if logical_start >= 0:
        sink = tl.load(metadata + 1)
        head = tl.load(metadata + 2)
        logical = logical_start + rows
        return tl.where(logical < sink, logical, sink + (head + logical - sink) % (capacity - sink))
    return rows


@triton.jit
def load_key(
    key,
    freqs,
    metadata,
    rows,
    columns,
    head,
    mask,
    capacity: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    logical_start: tl.constexpr,
    use_rope: tl.constexpr,
    rotary_dim: tl.constexpr,
    split_half: tl.constexpr,
    rope_fma: tl.constexpr,
    logical_visible_tokens: tl.constexpr = 0,
):
    physical = physical_rows(rows, metadata, capacity, logical_start)
    if logical_visible_tokens > 0:
        sink = tl.load(metadata + 1)
        recent_head = tl.load(metadata + 2)
        logical = tl.where(physical < sink, physical, sink + (physical - sink - recent_head + capacity - sink) % (capacity - sink))
        mask = mask & (logical[:, None] < logical_visible_tokens)
    addresses = (physical[:, None] * heads + head) * dim + columns[None, :]
    values = tl.load(key + addresses, mask, 0).to(tl.float32)
    if use_rope:
        sink = tl.load(metadata + 1)
        recent_head = tl.load(metadata + 2)
        logical = tl.where(physical < sink, physical, sink + (physical - sink - recent_head + capacity - sink) % (capacity - sink))
        if split_half:
            partner_column = tl.where(columns < rotary_dim // 2, columns + rotary_dim // 2, columns - rotary_dim // 2)
            frequency_column = columns % (rotary_dim // 2)
            first = columns < rotary_dim // 2
        else:
            partner_column = columns ^ 1
            frequency_column = columns // 2
            first = columns % 2 == 0
        rotated_mask = mask & (columns[None, :] < rotary_dim)
        partner = tl.load(key + (physical[:, None] * heads + head) * dim + partner_column[None, :], rotated_mask, 0).to(tl.float32)
        offset = logical[:, None] * rotary_dim + frequency_column[None, :]
        cosine = tl.load(freqs + offset, rotated_mask, 0).to(tl.float32)
        sine = tl.load(freqs + offset + rotary_dim // 2, rotated_mask, 0).to(tl.float32)
        product = partner * sine
        if rope_fma:
            rotated = tl.where(first[None, :], tl.fma(values, cosine, -product), tl.fma(values, cosine, product))
        else:
            rotated = tl.where(first[None, :], values * cosine - product, values * cosine + product)
        values = tl.where(columns[None, :] < rotary_dim, rotated.to(key.dtype.element_ty).to(tl.float32), values)
    return values


@triton.jit
def segment_is_written(metadata, part, write_tokens: tl.constexpr, capacity: tl.constexpr, segment_tokens: tl.constexpr):
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
    return sink_overlap | (recent_tokens > 0) & (first_overlap | second_overlap)


@triton.jit
def segment_is_dirty(metadata, valid, part, write_tokens: tl.constexpr, capacity: tl.constexpr, segment_tokens: tl.constexpr):
    return (tl.load(valid + part) == 0) | segment_is_written(metadata, part, write_tokens, capacity, segment_tokens)


@triton.jit
def segment_stats(
    key,
    value,
    freqs,
    metadata,
    valid,
    key_valid,
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
    logical_start: tl.constexpr = -1,
    use_rope: tl.constexpr = False,
    rotary_dim: tl.constexpr = 128,
    split_half: tl.constexpr = False,
    rope_fma: tl.constexpr = True,
    keys_only: tl.constexpr = False,
    logical_visible_tokens: tl.constexpr = 0,
    current_values: tl.constexpr = False,
    value_stride_token: tl.constexpr = 0,
    value_stride_head: tl.constexpr = 0,
    value_stride_dim: tl.constexpr = 0,
):
    part = tl.program_id(0)
    head = tl.program_id(1)
    dirty_key = logical_start >= 0 or segment_is_dirty(metadata, key_valid, part, write_tokens, capacity, segment_tokens)
    if current_values:
        dirty_value = not keys_only and segment_is_written(metadata, part, write_tokens, capacity, segment_tokens)
    else:
        dirty_value = not keys_only and (logical_start >= 0 or segment_is_dirty(metadata, valid, part, write_tokens, capacity, segment_tokens))
    if dirty_key or dirty_value:
        columns = tl.program_id(2) * columns_per_block + tl.arange(0, columns_per_block)
        rows = tl.arange(0, 64)
        length = tl.minimum(segment_tokens, visible_tokens - part * segment_tokens)
        key_sum = tl.full((columns_per_block,), 0, tl.float32)
        value_max = tl.full((columns_per_block,), 0, tl.float32)
        for offset in range(0, length, 64):
            local_rows = offset + rows
            mask = (local_rows[:, None] < length) & (columns[None, :] < dim)
            positions = part * segment_tokens + local_rows
            physical = physical_rows(positions, metadata, capacity, logical_start)
            addresses = (physical[:, None] * heads + head) * dim + columns[None, :]
            if dirty_key:
                keys = load_key(key, freqs, metadata, positions, columns, head, mask, capacity, heads, dim, logical_start, use_rope, rotary_dim, split_half, rope_fma, logical_visible_tokens)
                key_sum += tl.sum(keys, 0)
            if dirty_value:
                if current_values:
                    sink = tl.load(metadata + 1)
                    recent_head = tl.load(metadata + 2)
                    logical = tl.where(physical < sink, physical, sink + (physical - sink - recent_head + capacity - sink) % (capacity - sink))
                    current_rows = logical - tl.load(metadata)
                    addresses = current_rows[:, None] * value_stride_token + head * value_stride_head + columns[None, :] * value_stride_dim
                    mask = mask & (current_rows[:, None] >= 0) & (current_rows[:, None] < write_tokens)
                values = tl.load(value + addresses, mask, 0).to(tl.float32)
                value_max = tl.maximum(value_max, tl.max(tl.abs(values), 0))
        address = (part * heads + head) * dim + columns
        if dirty_key:
            tl.store(key_mean + address, key_sum / length, columns < dim)
        if dirty_value:
            tl.store(value_scale + address, tl.maximum(value_max, 1.0e-12) / scale_max, columns < dim)


@triton.jit
def quantize_key(
    key,
    freqs,
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
    logical_start: tl.constexpr = -1,
    use_rope: tl.constexpr = False,
    rotary_dim: tl.constexpr = 128,
    split_half: tl.constexpr = False,
    rope_fma: tl.constexpr = True,
    logical_visible_tokens: tl.constexpr = 0,
):
    part = tl.program_id(0)
    head = tl.program_id(1)
    group = tl.program_id(2)
    if logical_start >= 0 or segment_is_dirty(metadata, valid, part, write_tokens, capacity, segment_tokens):
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
            x0 = load_key(
                key,
                freqs,
                metadata,
                part * segment_tokens + rows0,
                columns,
                head,
                rows0[:, None] < length,
                capacity,
                heads,
                dim,
                logical_start,
                use_rope,
                rotary_dim,
                split_half,
                rope_fma,
                logical_visible_tokens,
            )
            x1 = load_key(
                key,
                freqs,
                metadata,
                part * segment_tokens + rows1,
                columns,
                head,
                rows1[:, None] < length,
                capacity,
                heads,
                dim,
                logical_start,
                use_rope,
                rotary_dim,
                split_half,
                rope_fma,
                logical_visible_tokens,
            )
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
            x = load_key(
                key,
                freqs,
                metadata,
                part * segment_tokens + rows,
                columns,
                head,
                rows[:, None] < length,
                capacity,
                heads,
                dim,
                logical_start,
                use_rope,
                rotary_dim,
                split_half,
                rope_fma,
                logical_visible_tokens,
            ) - center[None, :].to(tl.float32)
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
    logical_start: tl.constexpr = -1,
):
    part = tl.program_id(0)
    head = tl.program_id(1)
    if logical_start >= 0 or segment_is_dirty(metadata, valid, part, write_tokens, capacity, segment_tokens):
        rows = tl.program_id(2) // triton.cdiv(dim, 32) * 32 + tl.arange(0, 32)
        columns = tl.program_id(2) % triton.cdiv(dim, 32) * 32 + tl.arange(0, 32)
        length = tl.minimum(segment_tokens, visible_tokens - part * segment_tokens)
        physical = physical_rows(part * segment_tokens + rows, metadata, capacity, logical_start)
        address = (physical[:, None] * heads + head) * dim + columns[None, :]
        x = tl.load(value + address, (rows[:, None] < length) & (columns[None, :] < dim), 0).to(tl.float32)
        scale = tl.load(value_scale + (part * heads + head) * dim + columns, columns < dim, 1.0)
        maximum = tl.maximum((scale * scale_max).to(value.dtype.element_ty).to(tl.float32), 1.0e-12)
        inverse_scale = scale_max / maximum
        quantized = x * inverse_scale[None, :]
        permuted = (rows & ~15) | (rows & 1) | ((rows & 8) >> 2) | ((rows & 2) << 1) | ((rows & 4) << 1)
        output = ((part * dim + columns[None, :]) * heads + head) * padded_tokens + permuted[:, None]
        tl.store(value_fp8 + output, quantized, (rows[:, None] < padded_tokens) & (columns[None, :] < dim))


@triton.jit
def mark_valid(valid, key_valid, parts: tl.constexpr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    tl.store(valid + offsets, 1, offsets < parts)
    tl.store(key_valid + offsets, 1, offsets < parts)


@triton.jit
def cache_value_fp16(
    value,
    metadata,
    valid,
    value_cache,
    write_tokens: tl.constexpr,
    capacity: tl.constexpr,
    visible_tokens: tl.constexpr,
    segment_tokens: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    logical_start: tl.constexpr = -1,
):
    part = tl.program_id(0)
    head = tl.program_id(1)
    if logical_start >= 0 or segment_is_dirty(metadata, valid, part, write_tokens, capacity, segment_tokens):
        rows = tl.program_id(2) * 32 + tl.arange(0, 32)
        columns = tl.arange(0, dim)
        length = tl.minimum(segment_tokens, visible_tokens - part * segment_tokens)
        physical = physical_rows(part * segment_tokens + rows, metadata, capacity, logical_start)
        address = (physical[:, None] * heads + head) * dim + columns[None, :]
        x = tl.load(value + address, rows[:, None] < length, 0)
        output = ((part * segment_tokens + rows[:, None]) * heads + head) * dim + columns[None, :]
        tl.store(value_cache + output, x, rows[:, None] < segment_tokens)


@triton.jit
def cache_current_value(
    value,
    metadata,
    value_cache,
    value_scale,
    valid,
    write_tokens: tl.constexpr,
    capacity: tl.constexpr,
    segment_tokens: tl.constexpr,
    padded_tokens: tl.constexpr,
    scale_max: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    fp8_values: tl.constexpr,
    value_stride_token: tl.constexpr,
    value_stride_head: tl.constexpr,
    value_stride_dim: tl.constexpr,
    columns_per_block: tl.constexpr = 32,
):
    """Cache complete written segments directly, without retaining raw V."""
    current_part = tl.program_id(0)
    head = tl.program_id(1)
    row_block = tl.program_id(2) // triton.cdiv(dim, columns_per_block)
    column_block = tl.program_id(2) % triton.cdiv(dim, columns_per_block)
    columns = column_block * columns_per_block + tl.arange(0, columns_per_block)
    start = tl.load(metadata) + current_part * segment_tokens
    sink = tl.load(metadata + 1)
    recent_head = tl.load(metadata + 2)
    physical = tl.where(start < sink, start, sink + (recent_head + start - sink) % (capacity - sink))
    part = physical // segment_tokens
    local_rows = row_block * 32 + tl.arange(0, 32)
    source_rows = current_part * segment_tokens + local_rows
    address = source_rows[:, None] * value_stride_token + head * value_stride_head + columns[None, :] * value_stride_dim
    x = tl.load(value + address, (local_rows[:, None] < segment_tokens) & (columns[None, :] < dim), 0).to(tl.float32)
    if fp8_values:
        scale = tl.load(value_scale + (part * heads + head) * dim + columns, columns < dim, 1.0)
        # Match the native Sage value quantizer's input-dtype rounding exactly.
        maximum = tl.maximum((scale * scale_max).to(value.dtype.element_ty).to(tl.float32), 1.0e-12)
        inverse_scale = scale_max / maximum
        permuted = (local_rows & ~15) | (local_rows & 1) | ((local_rows & 8) >> 2) | ((local_rows & 2) << 1) | ((local_rows & 4) << 1)
        output = ((part * dim + columns[None, :]) * heads + head) * padded_tokens + permuted[:, None]
        tl.store(value_cache + output, x * inverse_scale[None, :], (local_rows[:, None] < padded_tokens) & (columns[None, :] < dim))
    else:
        output = ((part * segment_tokens + local_rows[:, None]) * heads + head) * dim + columns[None, :]
        tl.store(value_cache + output, x, (local_rows[:, None] < segment_tokens) & (columns[None, :] < dim))
    if tl.program_id(2) == 0:
        tl.store(valid + part, 1)


@triton.jit
def mark_keys_valid(key_valid, parts: tl.constexpr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    tl.store(key_valid + offsets, 1, offsets < parts)


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
    query_block: tl.constexpr = 4,
    combined_lse=None,
    return_lse: tl.constexpr = False,
):
    # The batched partial tensor can contain more than 2**31 elements even
    # when individual Q/K/V tensors fit comfortably in 32-bit indexing.
    row = (tl.program_id(0) * query_block + tl.arange(0, query_block)).to(tl.int64)
    head = tl.program_id(1)
    columns = tl.arange(0, dim)
    addresses = (row[:, None] * heads + head) * dim + columns[None, :]
    q = tl.load(query + addresses, row[:, None] < queries, 0).to(tl.float32)
    top = tl.full((query_block,), -float("inf"), tl.float32)
    denominator = tl.full((query_block,), 0, tl.float32)
    accumulator = tl.full((query_block, dim), 0, tl.float32)
    # Large windows can have hundreds of segments. Keep a compact loop rather
    # than compiling an entire copy of the reduction for every segment.
    for part in range(full_parts + has_tail):
        mean = tl.load(key_mean + (part * heads + head) * dim + columns).to(tl.float32)
        if part < full_parts:
            lse = tl.load(full_lse + (part * heads + head) * queries + row, row < queries, -float("inf"))
        else:
            lse = tl.load(tail_lse + head * queries + row, row < queries, -float("inf"))
        score = lse * 0.6931471805599453 + tl.sum(q * mean[None, :], 1) * dim**-0.5
        new_top = tl.maximum(top, score)
        old_weight = tl.exp(top - new_top)
        weight = tl.exp(score - new_top)
        partial_offset = part.to(tl.int64) * queries * heads * dim
        partial = tl.load(partial_output + (addresses + partial_offset), row[:, None] < queries, 0).to(tl.float32)
        accumulator = accumulator * old_weight[:, None] + partial * weight[:, None]
        denominator = denominator * old_weight + weight
        top = new_top
    tl.store(output + addresses, accumulator / denominator[:, None], row[:, None] < queries)
    if return_lse:
        tl.store(combined_lse + row * heads + head, top + tl.log(denominator), row < queries)
