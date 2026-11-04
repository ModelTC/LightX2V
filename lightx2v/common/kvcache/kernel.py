import torch
import triton
import triton.language as tl


@triton.jit
def _store_kv_ring(
    key,
    value,
    key_cache,
    value_cache,
    metadata,
    token_count: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    capacity: tl.constexpr,
    key_token_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    key_dim_stride: tl.constexpr,
    value_token_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    value_dim_stride: tl.constexpr,
    block_size: tl.constexpr,
    store_value: tl.constexpr = True,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < token_count * num_heads * head_dim
    token = offsets // (num_heads * head_dim)
    head = offsets // head_dim % num_heads
    feature = offsets % head_dim

    logical_start = tl.load(metadata)
    sink_tokens = tl.load(metadata + 1)
    recent_head = tl.load(metadata + 2)
    logical_token = logical_start + token
    physical_token = tl.where(
        logical_token < sink_tokens,
        logical_token,
        sink_tokens + (recent_head + logical_token - sink_tokens) % (capacity - sink_tokens),
    )
    key_offset = token * key_token_stride + head * key_head_stride + feature * key_dim_stride
    value_offset = token * value_token_stride + head * value_head_stride + feature * value_dim_stride
    cache_offset = (physical_token * num_heads + head) * head_dim + feature
    key_data = tl.load(key + key_offset, mask=valid)
    tl.store(key_cache + cache_offset, key_data, mask=valid)
    if store_value:
        value_data = tl.load(value + value_offset, mask=valid)
        tl.store(value_cache + cache_offset, value_data, mask=valid)


def store_kv_ring(key: torch.Tensor, value: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, metadata: torch.Tensor) -> None:
    """Store token-major KV using GPU ``[logical_start, sink_tokens, recent_head]``."""
    if torch.compiler.is_compiling():
        _store_kv_ring_op(key, value, key_cache, value_cache, metadata)
        return
    token_count, num_heads, head_dim = key.shape
    if token_count == 0:
        return
    block_size = 1024
    _store_kv_ring[(triton.cdiv(key.numel(), block_size),)](
        key,
        value,
        key_cache,
        value_cache,
        metadata,
        token_count,
        num_heads,
        head_dim,
        key_cache.shape[0],
        *key.stride(),
        *value.stride(),
        block_size,
    )


@torch.library.custom_op("lightx2v::store_kv_ring", mutates_args=("key_cache", "value_cache"), device_types="cuda")
def _store_kv_ring_op(key: torch.Tensor, value: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, metadata: torch.Tensor) -> None:
    store_kv_ring(key, value, key_cache, value_cache, metadata)


@_store_kv_ring_op.register_fake
def _store_kv_ring_fake(key, value, key_cache, value_cache, metadata):
    return None


def store_key_ring(key: torch.Tensor, key_cache: torch.Tensor, metadata: torch.Tensor) -> None:
    """Store raw K while the attention backend encodes current V directly."""
    if torch.compiler.is_compiling():
        _store_key_ring_op(key, key_cache, metadata)
        return
    if key.numel():
        _store_kv_ring[(triton.cdiv(key.numel(), 1024),)](
            key,
            key,
            key_cache,
            key_cache,
            metadata,
            *key.shape,
            key_cache.shape[0],
            *key.stride(),
            *key.stride(),
            1024,
            False,
        )


@torch.library.custom_op("lightx2v::store_key_ring", mutates_args=("key_cache",), device_types="cuda")
def _store_key_ring_op(key: torch.Tensor, key_cache: torch.Tensor, metadata: torch.Tensor) -> None:
    store_key_ring(key, key_cache, metadata)


@_store_key_ring_op.register_fake
def _store_key_ring_fake(key, key_cache, metadata):
    return None


@triton.jit
def _interleaved_rope(real, imag, cosine, sine):
    # Match FlashInfer's FP32 cosine FMA before the final FP16/BF16 cast.
    return tl.fma(real, cosine, -imag * sine), tl.fma(imag, cosine, real * sine)


@triton.jit
def _rotate_token_rope(
    tensor,
    output,
    freqs,
    metadata,
    count: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    token_stride: tl.constexpr,
    head_stride: tl.constexpr,
    dim_stride: tl.constexpr,
    offset: tl.constexpr,
    use_metadata: tl.constexpr,
    rotary_dim: tl.constexpr,
    split_half: tl.constexpr,
    rope_fma: tl.constexpr,
    block_pairs: tl.constexpr,
):
    offsets = tl.program_id(0) * block_pairs + tl.arange(0, block_pairs)
    valid = offsets < count
    token = offsets // (num_heads * head_dim)
    head = offsets // head_dim % num_heads
    feature = offsets % head_dim
    start = tl.load(metadata) if use_metadata else 0
    position = start + offset + token
    if split_half:
        partner = tl.where(feature < rotary_dim // 2, feature + rotary_dim // 2, feature - rotary_dim // 2)
        frequency = feature % (rotary_dim // 2)
        first = feature < rotary_dim // 2
    else:
        partner = feature ^ 1
        frequency = feature // 2
        first = feature % 2 == 0
    source = token * token_stride + head * head_stride
    values = tl.load(tensor + source + feature * dim_stride, mask=valid, other=0).to(tl.float32)
    rotated_mask = valid & (feature < rotary_dim)
    other = tl.load(tensor + source + partner * dim_stride, mask=rotated_mask, other=0).to(tl.float32)
    cosine = tl.load(freqs + position * rotary_dim + frequency, mask=rotated_mask, other=0).to(tl.float32)
    sine = tl.load(freqs + position * rotary_dim + rotary_dim // 2 + frequency, mask=rotated_mask, other=0).to(tl.float32)
    product = other * sine
    if rope_fma:
        rotated = tl.where(first, tl.fma(values, cosine, -product), tl.fma(values, cosine, product))
    else:
        rotated = tl.where(first, values * cosine - product, values * cosine + product)
    tl.store(output + offsets, tl.where(feature < rotary_dim, rotated, values), mask=valid)


@torch.library.custom_op("lightx2v::rotate_token_rope", mutates_args=(), device_types="cuda")
def rotate_token_rope(
    tensor: torch.Tensor,
    freqs: torch.Tensor,
    metadata: torch.Tensor | None = None,
    offset: int = 0,
    rotary_dim: int = 0,
    split_half: bool = False,
    rope_fma: bool = True,
) -> torch.Tensor:
    """Rotate a token view without gathering or modifying the raw KV storage.

    Frequencies are contiguous FP32 ``[tokens, cosine | sine]``. Query positions
    can read their current write start from GPU metadata during graph replay;
    physical key spans instead supply their logical starting offset.
    """
    output = torch.empty_like(tensor, memory_format=torch.contiguous_format)
    count = tensor.numel()
    if count:
        _rotate_token_rope[(triton.cdiv(count, 512),)](
            tensor,
            output,
            freqs,
            metadata,
            count,
            tensor.shape[1],
            tensor.shape[2],
            *tensor.stride(),
            offset,
            metadata is not None,
            rotary_dim or tensor.shape[2],
            split_half,
            rope_fma,
            512,
            enable_fp_fusion=rope_fma,
        )
    return output


@rotate_token_rope.register_fake
def _rotate_token_rope_fake(tensor, freqs, metadata=None, offset=0, rotary_dim=0, split_half=False, rope_fma=True):
    return torch.empty_like(tensor, memory_format=torch.contiguous_format)


@triton.jit
def _rotate_ring_qk(
    query,
    raw_key,
    rotated_key,
    freqs,
    metadata,
    pair_count: tl.constexpr,
    query_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    capacity: tl.constexpr,
    visible_tokens: tl.constexpr,
    query_token_stride: tl.constexpr,
    query_head_stride: tl.constexpr,
    query_dim_stride: tl.constexpr,
    block_pairs: tl.constexpr,
):
    pairs = tl.program_id(0) * block_pairs + tl.arange(0, block_pairs)
    valid = pairs < pair_count
    physical_token = pairs // (num_heads * (head_dim // 2))
    feature = pairs % (head_dim // 2)
    head = pairs // (head_dim // 2) % num_heads
    write_start = tl.load(metadata)
    sink_tokens = tl.load(metadata + 1)
    recent_head = tl.load(metadata + 2)
    logical_token = tl.where(
        physical_token < sink_tokens,
        physical_token,
        sink_tokens + (physical_token - sink_tokens - recent_head + capacity - sink_tokens) % (capacity - sink_tokens),
    )
    visible = valid & (logical_token < visible_tokens)
    cosine = tl.load(freqs + logical_token * head_dim + feature, mask=visible, other=0)
    sine = tl.load(freqs + logical_token * head_dim + head_dim // 2 + feature, mask=visible, other=0)
    key_real = tl.load(raw_key + pairs * 2, mask=valid, other=0).to(tl.float32)
    key_imag = tl.load(raw_key + pairs * 2 + 1, mask=valid, other=0).to(tl.float32)
    key_real, key_imag = _interleaved_rope(key_real, key_imag, cosine, sine)
    tl.store(rotated_key + pairs * 2, key_real, mask=valid)
    tl.store(rotated_key + pairs * 2 + 1, key_imag, mask=valid)
    query_token = logical_token - write_start
    current = visible & (query_token >= 0) & (query_token < query_tokens)
    query_offset = query_token * query_token_stride + head * query_head_stride + feature * 2 * query_dim_stride
    query_real = tl.load(query + query_offset, mask=current, other=0).to(tl.float32)
    query_imag = tl.load(query + query_offset + query_dim_stride, mask=current, other=0).to(tl.float32)
    query_real, query_imag = _interleaved_rope(query_real, query_imag, cosine, sine)
    tl.store(query + query_offset, query_real, mask=current)
    tl.store(query + query_offset + query_dim_stride, query_imag, mask=current)


@triton.jit
def _rotate_ring_qk_partial(
    query,
    raw_key,
    rotated_key,
    freqs,
    metadata,
    count: tl.constexpr,
    query_tokens: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    capacity: tl.constexpr,
    visible: tl.constexpr,
    query_token_stride: tl.constexpr,
    query_head_stride: tl.constexpr,
    query_dim_stride: tl.constexpr,
    rotary_dim: tl.constexpr,
    split_half: tl.constexpr,
    rope_fma: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < count
    physical = offsets // (heads * dim)
    head = offsets // dim % heads
    column = offsets % dim
    start, sink, recent = tl.load(metadata), tl.load(metadata + 1), tl.load(metadata + 2)
    logical = tl.where(physical < sink, physical, sink + (physical - sink - recent + capacity - sink) % (capacity - sink))
    if split_half:
        partner = tl.where(column < rotary_dim // 2, column + rotary_dim // 2, column - rotary_dim // 2)
        frequency = column % (rotary_dim // 2)
        first = column < rotary_dim // 2
    else:
        partner = column ^ 1
        frequency = column // 2
        first = column % 2 == 0
    rotate = valid & (logical < visible) & (column < rotary_dim)
    cosine = tl.load(freqs + logical * rotary_dim + frequency, rotate, 0).to(tl.float32)
    sine = tl.load(freqs + logical * rotary_dim + rotary_dim // 2 + frequency, rotate, 0).to(tl.float32)
    key = tl.load(raw_key + offsets, valid, 0).to(tl.float32)
    other = tl.load(raw_key + (physical * heads + head) * dim + partner, rotate, 0).to(tl.float32)
    if rope_fma:
        rotated = tl.where(first, tl.fma(key, cosine, -other * sine), tl.fma(key, cosine, other * sine))
    else:
        rotated = tl.where(first, key * cosine - other * sine, key * cosine + other * sine)
    tl.store(rotated_key + offsets, tl.where(column < rotary_dim, rotated, key), valid)
    query_token = logical - start
    current = valid & (logical < visible) & (query_token >= 0) & (query_token < query_tokens)
    address = query_token * query_token_stride + head * query_head_stride
    q = tl.load(query + address + column * query_dim_stride, current, 0).to(tl.float32)
    q_other = tl.load(query + address + partner * query_dim_stride, current & (column < rotary_dim), 0).to(tl.float32)
    if rope_fma:
        q_rotated = tl.where(first, tl.fma(q, cosine, -q_other * sine), tl.fma(q, cosine, q_other * sine))
    else:
        q_rotated = tl.where(first, q * cosine - q_other * sine, q * cosine + q_other * sine)
    tl.store(query + address + column * query_dim_stride, tl.where(column < rotary_dim, q_rotated, q), current)


def rotate_ring_qk(
    query: torch.Tensor,
    raw_key: torch.Tensor,
    rotated_key: torch.Tensor,
    freqs: torch.Tensor,
    metadata: torch.Tensor,
    *,
    rotary_dim=None,
    rope_layout="interleaved",
    rope_fma=True,
) -> None:
    """Rotate Q and physical-order K using each slot's current logical position."""
    if torch.compiler.is_compiling():
        _rotate_ring_qk_op(query, raw_key, rotated_key, freqs, metadata, rotary_dim or raw_key.shape[-1], rope_layout == "split_half", rope_fma)
        return
    rotary_dim = rotary_dim or raw_key.shape[-1]
    if rotary_dim != raw_key.shape[-1] or rope_layout == "split_half" or not rope_fma:
        # Split-half partners can live in different warps. Produce Q out of
        # place before copying back, so one warp cannot overwrite another
        # warp's input while it is still reading its rotation partner.
        if query.numel():
            query.copy_(rotate_token_rope(query, freqs, metadata, rotary_dim=rotary_dim, split_half=rope_layout == "split_half", rope_fma=rope_fma))
        _rotate_ring_qk_partial[(triton.cdiv(rotated_key.numel(), 512),)](
            query,
            raw_key,
            rotated_key,
            freqs,
            metadata,
            rotated_key.numel(),
            0,
            raw_key.shape[1],
            raw_key.shape[2],
            raw_key.shape[0],
            freqs.shape[0],
            *query.stride(),
            rotary_dim,
            rope_layout == "split_half",
            rope_fma,
            512,
            enable_fp_fusion=rope_fma,
        )
        return
    pairs = rotated_key.numel() // 2
    block_pairs = 512
    _rotate_ring_qk[(triton.cdiv(pairs, block_pairs),)](
        query,
        raw_key,
        rotated_key,
        freqs,
        metadata,
        pairs,
        query.shape[0],
        raw_key.shape[1],
        raw_key.shape[2],
        raw_key.shape[0],
        freqs.shape[0],
        *query.stride(),
        block_pairs,
    )


@torch.library.custom_op("lightx2v::rotate_ring_qk", mutates_args=("query", "rotated_key"), device_types="cuda")
def _rotate_ring_qk_op(
    query: torch.Tensor,
    raw_key: torch.Tensor,
    rotated_key: torch.Tensor,
    freqs: torch.Tensor,
    metadata: torch.Tensor,
    rotary_dim: int,
    split_half: bool,
    rope_fma: bool,
) -> None:
    rotate_ring_qk(query, raw_key, rotated_key, freqs, metadata, rotary_dim=rotary_dim, rope_layout="split_half" if split_half else "interleaved", rope_fma=rope_fma)


@_rotate_ring_qk_op.register_fake
def _rotate_ring_qk_fake(query, raw_key, rotated_key, freqs, metadata, rotary_dim, split_half, rope_fma):
    return None
