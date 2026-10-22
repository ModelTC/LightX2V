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
    value_data = tl.load(value + value_offset, mask=valid)
    tl.store(key_cache + cache_offset, key_data, mask=valid)
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


@triton.jit
def _interleaved_rope(real, imag, cosine, sine):
    # Match FlashInfer's FP32 cosine FMA before the final FP16/BF16 cast.
    return tl.fma(real, cosine, -imag * sine), tl.fma(imag, cosine, real * sine)


@triton.jit
def _gather_kv_cache_rope(
    key_cache,
    value_cache,
    key_out,
    value_out,
    freqs,
    spans,
    pair_count: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    span_count: tl.constexpr,
    block_pairs: tl.constexpr,
):
    pairs = tl.program_id(0) * block_pairs + tl.arange(0, block_pairs)
    valid = pairs < pair_count
    token = pairs // (num_heads * (head_dim // 2))
    feature = pairs % (head_dim // 2)
    head = pairs // (head_dim // 2) % num_heads
    source_token = tl.full((block_pairs,), 0, tl.int32)
    for span in tl.static_range(span_count):
        physical_start = tl.load(spans + span * 3)
        output_start = tl.load(spans + span * 3 + 1)
        length = tl.load(spans + span * 3 + 2)
        in_span = (token >= output_start) & (token < output_start + length)
        source_token = tl.where(in_span, physical_start + token - output_start, source_token)
    source = (source_token * num_heads + head) * head_dim + feature * 2
    real = tl.load(key_cache + source, mask=valid, other=0).to(tl.float32)
    imag = tl.load(key_cache + source + 1, mask=valid, other=0).to(tl.float32)
    cosine = tl.load(freqs + token * head_dim + feature, mask=valid, other=0)
    sine = tl.load(freqs + token * head_dim + head_dim // 2 + feature, mask=valid, other=0)
    rotated_real, rotated_imag = _interleaved_rope(real, imag, cosine, sine)
    value_real = tl.load(value_cache + source, mask=valid, other=0)
    value_imag = tl.load(value_cache + source + 1, mask=valid, other=0)
    tl.store(key_out + pairs * 2, rotated_real, mask=valid)
    tl.store(key_out + pairs * 2 + 1, rotated_imag, mask=valid)
    tl.store(value_out + pairs * 2, value_real, mask=valid)
    tl.store(value_out + pairs * 2 + 1, value_imag, mask=valid)


def gather_kv_cache_rope(key_cache: torch.Tensor, value_cache: torch.Tensor, key_out: torch.Tensor, value_out: torch.Tensor, freqs: torch.Tensor, spans: torch.Tensor) -> None:
    """Gather raw ring KV into ordered workspace, rotating only K.

    Buffers use contiguous [tokens, heads, width] storage. ``spans`` is a CUDA
    int32 table of (physical_start, output_start, length); zero-length rows pad
    the table. Frequencies are prepared FP32 [cos | sin] in output token order.
    """
    if torch.compiler.is_compiling():
        _gather_kv_cache_rope_op(key_cache, value_cache, key_out, value_out, freqs, spans)
        return
    if key_out.numel() == 0:
        return
    if key_out.shape[-1] % 2:
        raise ValueError("Interleaved RoPE requires an even head dimension")
    block_pairs = 512
    pair_count = key_out.numel() // 2
    _gather_kv_cache_rope[(triton.cdiv(pair_count, block_pairs),)](
        key_cache,
        value_cache,
        key_out,
        value_out,
        freqs,
        spans,
        pair_count,
        key_out.shape[1],
        key_out.shape[2],
        spans.shape[0],
        block_pairs,
    )


@torch.library.custom_op("lightx2v::gather_kv_cache_rope", mutates_args=("key_out", "value_out"), device_types="cuda")
def _gather_kv_cache_rope_op(key_cache: torch.Tensor, value_cache: torch.Tensor, key_out: torch.Tensor, value_out: torch.Tensor, freqs: torch.Tensor, spans: torch.Tensor) -> None:
    gather_kv_cache_rope(key_cache, value_cache, key_out, value_out, freqs, spans)


@_gather_kv_cache_rope_op.register_fake
def _gather_kv_cache_rope_fake(key_cache, value_cache, key_out, value_out, freqs, spans):
    return None


@triton.jit
def _update_kv_cache_rope(
    query,
    key,
    value,
    key_out,
    value_out,
    freqs,
    pair_count: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    query_token_stride: tl.constexpr,
    query_head_stride: tl.constexpr,
    query_dim_stride: tl.constexpr,
    key_token_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    key_dim_stride: tl.constexpr,
    value_token_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    value_dim_stride: tl.constexpr,
    block_pairs: tl.constexpr,
):
    pairs = tl.program_id(0) * block_pairs + tl.arange(0, block_pairs)
    valid = pairs < pair_count
    token = pairs // (num_heads * (head_dim // 2))
    feature = pairs % (head_dim // 2)
    head = pairs // (head_dim // 2) % num_heads
    query_offset = token * query_token_stride + head * query_head_stride + feature * 2 * query_dim_stride
    key_offset = token * key_token_stride + head * key_head_stride + feature * 2 * key_dim_stride
    value_offset = token * value_token_stride + head * value_head_stride + feature * 2 * value_dim_stride
    cosine = tl.load(freqs + token * head_dim + feature, mask=valid, other=0)
    sine = tl.load(freqs + token * head_dim + head_dim // 2 + feature, mask=valid, other=0)
    query_real = tl.load(query + query_offset, mask=valid, other=0).to(tl.float32)
    query_imag = tl.load(query + query_offset + query_dim_stride, mask=valid, other=0).to(tl.float32)
    key_real = tl.load(key + key_offset, mask=valid, other=0).to(tl.float32)
    key_imag = tl.load(key + key_offset + key_dim_stride, mask=valid, other=0).to(tl.float32)
    rotated_query_real, rotated_query_imag = _interleaved_rope(query_real, query_imag, cosine, sine)
    rotated_key_real, rotated_key_imag = _interleaved_rope(key_real, key_imag, cosine, sine)
    value_real = tl.load(value + value_offset, mask=valid, other=0)
    value_imag = tl.load(value + value_offset + value_dim_stride, mask=valid, other=0)
    tl.store(query + query_offset, rotated_query_real, mask=valid)
    tl.store(query + query_offset + query_dim_stride, rotated_query_imag, mask=valid)
    tl.store(key_out + pairs * 2, rotated_key_real, mask=valid)
    tl.store(key_out + pairs * 2 + 1, rotated_key_imag, mask=valid)
    tl.store(value_out + pairs * 2, value_real, mask=valid)
    tl.store(value_out + pairs * 2 + 1, value_imag, mask=valid)


def update_kv_cache_rope(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_out: torch.Tensor, value_out: torch.Tensor, freqs: torch.Tensor) -> None:
    """Rotate Q in place and write rotated K / unchanged V into workspace.

    Q/K/V have matching [tokens, heads, width] shapes and may be strided.
    Raw K/V are preserved for commit; output buffers are contiguous tail views.
    Frequencies are prepared FP32 [cos | sin], using interleaved feature pairs.
    """
    if torch.compiler.is_compiling():
        _update_kv_cache_rope_op(query, key, value, key_out, value_out, freqs)
        return
    if key.numel() == 0:
        return
    if key.shape[-1] % 2:
        raise ValueError("Interleaved RoPE requires an even head dimension")
    block_pairs = 512
    pair_count = key.numel() // 2
    _update_kv_cache_rope[(triton.cdiv(pair_count, block_pairs),)](
        query,
        key,
        value,
        key_out,
        value_out,
        freqs,
        pair_count,
        key.shape[1],
        key.shape[2],
        *query.stride(),
        *key.stride(),
        *value.stride(),
        block_pairs,
    )


@torch.library.custom_op("lightx2v::update_kv_cache_rope", mutates_args=("query", "key_out", "value_out"), device_types="cuda")
def _update_kv_cache_rope_op(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_out: torch.Tensor, value_out: torch.Tensor, freqs: torch.Tensor) -> None:
    update_kv_cache_rope(query, key, value, key_out, value_out, freqs)


@_update_kv_cache_rope_op.register_fake
def _update_kv_cache_rope_fake(query, key, value, key_out, value_out, freqs):
    return None


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
    cosine = tl.load(freqs + logical_token * head_dim + feature, mask=valid, other=0)
    sine = tl.load(freqs + logical_token * head_dim + head_dim // 2 + feature, mask=valid, other=0)
    key_real = tl.load(raw_key + pairs * 2, mask=valid, other=0).to(tl.float32)
    key_imag = tl.load(raw_key + pairs * 2 + 1, mask=valid, other=0).to(tl.float32)
    key_real, key_imag = _interleaved_rope(key_real, key_imag, cosine, sine)
    tl.store(rotated_key + pairs * 2, key_real, mask=valid)
    tl.store(rotated_key + pairs * 2 + 1, key_imag, mask=valid)
    query_token = logical_token - write_start
    current = valid & (query_token >= 0) & (query_token < query_tokens)
    query_offset = query_token * query_token_stride + head * query_head_stride + feature * 2 * query_dim_stride
    query_real = tl.load(query + query_offset, mask=current, other=0).to(tl.float32)
    query_imag = tl.load(query + query_offset + query_dim_stride, mask=current, other=0).to(tl.float32)
    query_real, query_imag = _interleaved_rope(query_real, query_imag, cosine, sine)
    tl.store(query + query_offset, query_real, mask=current)
    tl.store(query + query_offset + query_dim_stride, query_imag, mask=current)


def rotate_ring_qk(query: torch.Tensor, raw_key: torch.Tensor, rotated_key: torch.Tensor, freqs: torch.Tensor, metadata: torch.Tensor) -> None:
    """Rotate Q and physical-order K using each slot's current logical position."""
    if torch.compiler.is_compiling():
        _rotate_ring_qk_op(query, raw_key, rotated_key, freqs, metadata)
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
        *query.stride(),
        block_pairs,
    )


@torch.library.custom_op("lightx2v::rotate_ring_qk", mutates_args=("query", "rotated_key"), device_types="cuda")
def _rotate_ring_qk_op(query: torch.Tensor, raw_key: torch.Tensor, rotated_key: torch.Tensor, freqs: torch.Tensor, metadata: torch.Tensor) -> None:
    rotate_ring_qk(query, raw_key, rotated_key, freqs, metadata)


@_rotate_ring_qk_op.register_fake
def _rotate_ring_qk_fake(query, raw_key, rotated_key, freqs, metadata):
    return None
