"""Persistent, incrementally quantized physical KV segments for SageAttention2."""

from dataclasses import replace
from functools import lru_cache
from typing import NamedTuple

import torch
import triton

from .kernels.sage_ring import cache_current_value, cache_value_fp16, mark_keys_valid, mark_valid, merge_attention, quantize_key, quantize_value, segment_stats
from .ring import RingAttention

# Ampere's native PV kernel consumes FP16; Ada, Hopper and SM120 consume FP8.
_ARCH_CONFIG = {80: (64, True, 1.0), 86: (64, True, 1.0), 89: (64, True, 2.25), 90: (128, True, 448.0), 120: (64, False, 2.25)}


def sage_value_dtype(architecture):
    if architecture not in _ARCH_CONFIG:
        raise ValueError(f"Sage KV caching does not support SM{architecture}")
    return torch.float16 if architecture in (80, 86) else torch.float8_e4m3fn


class SageKVCache(NamedTuple):
    key_int8: torch.Tensor
    key_scale: torch.Tensor
    key_mean: torch.Tensor
    value_fp8: torch.Tensor
    value_scale: torch.Tensor
    valid: torch.Tensor
    key_valid: torch.Tensor
    architecture: int

    @property
    def bindings(self) -> tuple[torch.Tensor, ...]:
        return self.key_int8, self.key_scale, self.key_mean, self.value_fp8, self.value_scale, self.valid, self.key_valid

    @property
    def value(self) -> torch.Tensor:
        """The only resident V tensor, in the architecture's native PV dtype."""
        return self.value_fp8


def allocate_sage_kv_cache(key_ring: torch.Tensor, segment_tokens: int, *, capacity=None) -> SageKVCache:
    """Allocate a quantized companion of a resident [tokens, heads, dim] KV ring."""
    major, minor = torch.cuda.get_device_capability(key_ring.device)
    architecture = major * 10 + minor
    if architecture not in _ARCH_CONFIG:
        raise ValueError("SageAttention2 ring caching requires SM80/86, SM89, SM90, or SM120")
    ring_capacity, heads, dim = key_ring.shape
    capacity = ring_capacity if capacity is None else capacity
    if dim not in (64, 128) or not 0 < segment_tokens <= capacity:
        raise ValueError("SageAttention2 ring caching requires head_dim 64/128 and a nonempty segment within capacity")
    key_block, per_thread, _ = _ARCH_CONFIG[architecture]
    parts = triton.cdiv(capacity, segment_tokens)
    scales = triton.cdiv(segment_tokens, key_block) * (4 if per_thread else 1)
    padded_tokens = triton.cdiv(segment_tokens, 128 if architecture == 90 else 64) * (128 if architecture == 90 else 64)
    value_dtype = sage_value_dtype(architecture)
    value_shape = (parts, segment_tokens, heads, dim) if value_dtype == torch.float16 else (parts, dim, heads, padded_tokens)
    return SageKVCache(
        key_ring.new_empty((parts, segment_tokens, heads, dim), dtype=torch.int8),
        key_ring.new_empty((parts, heads, scales), dtype=torch.float32),
        key_ring.new_empty((parts, heads, dim)),
        key_ring.new_empty(value_shape, dtype=value_dtype),
        key_ring.new_empty((0,) if value_dtype == torch.float16 else (parts, heads, dim), dtype=torch.float32),
        torch.zeros(parts, device=key_ring.device, dtype=torch.int32),
        torch.zeros(parts, device=key_ring.device, dtype=torch.int32),
        architecture,
    )


def update_sage_kv_cache(
    key_ring,
    value_ring,
    metadata,
    state: SageKVCache,
    write_tokens: int,
    visible_tokens: int,
    *,
    freqs=None,
    logical_start=-1,
    rotary_dim=None,
    rope_layout="interleaved",
    rope_fma=True,
) -> None:
    """Requantize initialized or newly written segments after the ring slot write."""
    rotary_dim = key_ring.shape[-1] if rotary_dim is None else rotary_dim
    if freqs is not None:
        if rope_layout not in ("interleaved", "split_half") or not 0 < rotary_dim <= key_ring.shape[-1] or rotary_dim % 2:
            raise ValueError("Sage RoPE requires an even rotary_dim within head_dim and an interleaved or split_half layout")
        if freqs.ndim != 2 or freqs.shape[-1] != rotary_dim or not freqs.is_contiguous():
            raise ValueError("Sage RoPE frequencies must be contiguous [tokens, cosine | sine] with width rotary_dim")
    _update_sage_kv_cache(
        key_ring,
        value_ring,
        metadata,
        *state.bindings,
        key_ring if freqs is None else freqs,
        write_tokens,
        visible_tokens,
        state.architecture,
        logical_start,
        freqs is not None,
        rotary_dim,
        rope_layout == "split_half",
        rope_fma,
    )


@torch.library.custom_op(
    "lightx2v::update_sage_kv_cache",
    mutates_args=("key_int8", "key_scale", "key_mean", "value_fp8", "value_scale", "valid", "key_valid"),
    device_types="cuda",
)
def _update_sage_kv_cache(
    key_ring: torch.Tensor,
    value_ring: torch.Tensor,
    metadata: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_mean: torch.Tensor,
    value_fp8: torch.Tensor,
    value_scale: torch.Tensor,
    valid: torch.Tensor,
    key_valid: torch.Tensor,
    freqs: torch.Tensor,
    write_tokens: int,
    visible_tokens: int,
    architecture: int,
    logical_start: int,
    use_rope: bool,
    rotary_dim: int,
    split_half: bool,
    rope_fma: bool,
) -> None:
    key_block, per_thread, scale_max = _ARCH_CONFIG[architecture]
    _, segment_tokens, heads, dim = key_int8.shape
    parts = triton.cdiv(visible_tokens, segment_tokens)
    capacity = key_ring.shape[0]
    segment_stats[(parts, heads, triton.cdiv(dim, 32))](
        key_ring,
        value_ring,
        freqs,
        metadata,
        valid,
        key_valid,
        key_mean,
        value_scale,
        write_tokens,
        capacity,
        visible_tokens,
        segment_tokens,
        heads,
        dim,
        scale_max,
        32,
        logical_start,
        use_rope,
        rotary_dim,
        split_half,
        rope_fma,
        architecture in (80, 86),
        num_warps=4,
        enable_fp_fusion=rope_fma,
    )
    quantize_key[(parts, heads, key_scale.shape[-1])](
        key_ring,
        freqs,
        metadata,
        key_valid,
        key_mean,
        key_int8,
        key_scale,
        write_tokens,
        capacity,
        visible_tokens,
        segment_tokens,
        heads,
        dim,
        key_block,
        per_thread,
        key_scale.shape[-1],
        logical_start,
        use_rope,
        rotary_dim,
        split_half,
        rope_fma,
        num_warps=4,
        enable_fp_fusion=rope_fma,
    )
    if architecture in (80, 86):
        cache_value_fp16[(parts, heads, triton.cdiv(segment_tokens, 32))](
            value_ring, metadata, valid, value_fp8, write_tokens, capacity, visible_tokens, segment_tokens, heads, dim, logical_start, num_warps=4
        )
    else:
        quantize_value[(parts, heads, triton.cdiv(value_fp8.shape[-1], 32) * triton.cdiv(dim, 32))](
            value_ring,
            metadata,
            valid,
            value_scale,
            value_fp8,
            write_tokens,
            capacity,
            visible_tokens,
            segment_tokens,
            value_fp8.shape[-1],
            scale_max,
            heads,
            dim,
            logical_start,
            num_warps=4,
        )
    mark_valid[(1,)](valid, key_valid, parts, triton.next_power_of_2(parts))


@_update_sage_kv_cache.register_fake
def _update_sage_kv_cache_fake(
    key_ring,
    value_ring,
    metadata,
    key_int8,
    key_scale,
    key_mean,
    value_fp8,
    value_scale,
    valid,
    key_valid,
    freqs,
    write_tokens,
    visible_tokens,
    architecture,
    logical_start,
    use_rope,
    rotary_dim,
    split_half,
    rope_fma,
):
    return None


def update_sage_kv_cache_from_current(
    key_ring,
    current_value,
    metadata,
    state: SageKVCache,
    write_tokens: int,
    visible_tokens: int,
    *,
    physical_tokens=None,
    freqs=None,
    logical_start=-1,
    rotary_dim=None,
    rope_layout="interleaved",
    rope_fma=True,
):
    """Refresh raw K and cache current V directly in complete physical segments.

    Historical V exists only in ``state.value``. Geometry must keep writes and
    the sink aligned to complete segments, so V statistics never need to read
    a previously quantized value to rebuild a partially overwritten segment.
    """
    segment_tokens = state.key_int8.shape[1]
    if logical_start != -1:
        raise ValueError("V-only caching requires physical segments; select visible quantized spans at attention time")
    physical_tokens = visible_tokens if physical_tokens is None else physical_tokens
    if key_ring.shape[0] % segment_tokens or write_tokens % segment_tokens or visible_tokens % segment_tokens or physical_tokens % segment_tokens:
        raise ValueError("V-only caching requires complete aligned segment writes")
    if not visible_tokens <= physical_tokens <= key_ring.shape[0]:
        raise ValueError("physical_tokens must cover the visible window within the raw K capacity")
    if current_value.shape != (write_tokens, *key_ring.shape[1:]):
        raise ValueError("current V must have shape [write_tokens, heads, dim]")
    rotary_dim = key_ring.shape[-1] if rotary_dim is None else rotary_dim
    if freqs is not None:
        if rope_layout not in ("interleaved", "split_half") or not 0 < rotary_dim <= key_ring.shape[-1] or rotary_dim % 2:
            raise ValueError("Sage RoPE requires an even rotary_dim and an interleaved or split_half layout")
        if freqs.ndim != 2 or freqs.shape[-1] != rotary_dim or not freqs.is_contiguous():
            raise ValueError("Sage RoPE frequencies must be contiguous [tokens, cosine | sine]")
    _update_sage_kv_cache_from_current(
        key_ring,
        current_value,
        metadata,
        *state.bindings,
        key_ring if freqs is None else freqs,
        write_tokens,
        visible_tokens,
        physical_tokens,
        state.architecture,
        freqs is not None,
        rotary_dim,
        rope_layout == "split_half",
        rope_fma,
    )


@torch.library.custom_op(
    "lightx2v::update_sage_kv_cache_from_current",
    mutates_args=("key_int8", "key_scale", "key_mean", "value_fp8", "value_scale", "valid", "key_valid"),
    device_types="cuda",
)
def _update_sage_kv_cache_from_current(
    key_ring: torch.Tensor,
    current_value: torch.Tensor,
    metadata: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_mean: torch.Tensor,
    value_fp8: torch.Tensor,
    value_scale: torch.Tensor,
    valid: torch.Tensor,
    key_valid: torch.Tensor,
    freqs: torch.Tensor,
    write_tokens: int,
    visible_tokens: int,
    physical_tokens: int,
    architecture: int,
    use_rope: bool,
    rotary_dim: int,
    split_half: bool,
    rope_fma: bool,
) -> None:
    key_block, per_thread, scale_max = _ARCH_CONFIG[architecture]
    _, segment_tokens, heads, dim = key_int8.shape
    capacity = key_ring.shape[0]
    parts = triton.cdiv(physical_tokens, segment_tokens)
    logical_visible_tokens = visible_tokens if physical_tokens != visible_tokens else 0
    segment_stats[(parts, heads, triton.cdiv(dim, 32))](
        key_ring,
        current_value,
        freqs,
        metadata,
        valid,
        key_valid,
        key_mean,
        value_scale,
        write_tokens,
        capacity,
        physical_tokens,
        segment_tokens,
        heads,
        dim,
        scale_max,
        32,
        -1,
        use_rope,
        rotary_dim,
        split_half,
        rope_fma,
        architecture in (80, 86),
        logical_visible_tokens,
        True,
        *current_value.stride(),
        num_warps=4,
        enable_fp_fusion=rope_fma,
    )
    quantize_key[(parts, heads, key_scale.shape[-1])](
        key_ring,
        freqs,
        metadata,
        key_valid,
        key_mean,
        key_int8,
        key_scale,
        write_tokens,
        capacity,
        physical_tokens,
        segment_tokens,
        heads,
        dim,
        key_block,
        per_thread,
        key_scale.shape[-1],
        -1,
        use_rope,
        rotary_dim,
        split_half,
        rope_fma,
        logical_visible_tokens,
        num_warps=4,
        enable_fp_fusion=rope_fma,
    )
    fp8_values = architecture not in (80, 86)
    value_rows = value_fp8.shape[-1] if fp8_values else segment_tokens
    cache_current_value[(write_tokens // segment_tokens, heads, triton.cdiv(value_rows, 32) * triton.cdiv(dim, 32))](
        current_value,
        metadata,
        value_fp8,
        value_scale,
        valid,
        write_tokens,
        capacity,
        segment_tokens,
        value_rows,
        scale_max,
        heads,
        dim,
        fp8_values,
        *current_value.stride(),
        num_warps=4,
        enable_fp_fusion=False,
    )
    mark_keys_valid[(1,)](key_valid, parts, triton.next_power_of_2(parts))


@_update_sage_kv_cache_from_current.register_fake
def _update_sage_kv_cache_from_current_fake(
    key_ring,
    current_value,
    metadata,
    key_int8,
    key_scale,
    key_mean,
    value_fp8,
    value_scale,
    valid,
    key_valid,
    freqs,
    write_tokens,
    visible_tokens,
    physical_tokens,
    architecture,
    use_rope,
    rotary_dim,
    split_half,
    rope_fma,
):
    return None


def encode_sage_current(key, value, metadata, state, capacity, *, cache_values=True, physical_start=-1, position=None, enabled_after=-1):
    """Write complete current segments directly into persistent encoded K/V."""
    segment_tokens = state.key_int8.shape[1]
    if key.ndim != 3 or key.shape[0] <= 0 or key.shape[0] % segment_tokens:
        raise ValueError("Encoded-only Sage requires nonempty whole-segment K writes")
    if cache_values and value.shape != key.shape:
        raise ValueError("Encoded-only Sage requires matching current K/V shapes")
    _encode_sage_current(
        key,
        value,
        metadata,
        metadata if position is None else position,
        *state.bindings,
        capacity,
        state.architecture,
        cache_values,
        physical_start,
        enabled_after,
    )


@torch.library.custom_op(
    "lightx2v::encode_sage_current",
    mutates_args=("key_int8", "key_scale", "key_mean", "value_fp8", "value_scale", "valid", "key_valid"),
    device_types="cuda",
)
def _encode_sage_current(
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: torch.Tensor,
    position: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_mean: torch.Tensor,
    value_fp8: torch.Tensor,
    value_scale: torch.Tensor,
    valid: torch.Tensor,
    key_valid: torch.Tensor,
    capacity: int,
    architecture: int,
    cache_values: bool,
    physical_start: int,
    enabled_after: int,
) -> None:
    from .kernels.sage_current import current_stats, quantize_current_key

    key_block, per_thread, scale_max = _ARCH_CONFIG[architecture]
    _, segment_tokens, heads, dim = key_int8.shape
    parts = key.shape[0] // segment_tokens
    fp8_values = architecture not in (80, 86)
    current_stats[(parts, heads, triton.cdiv(dim, 32))](
        key,
        value,
        metadata,
        position,
        key_mean,
        value_scale,
        capacity,
        segment_tokens,
        heads,
        dim,
        *key.stride(),
        *value.stride(),
        scale_max,
        cache_values and fp8_values,
        physical_start,
        enabled_after,
        num_warps=4,
        enable_fp_fusion=True,
    )
    quantize_current_key[(parts, heads, key_scale.shape[-1])](
        key,
        metadata,
        position,
        key_mean,
        key_int8,
        key_scale,
        key_valid,
        capacity,
        segment_tokens,
        heads,
        dim,
        *key.stride(),
        key_block,
        per_thread,
        key_scale.shape[-1],
        physical_start,
        enabled_after,
        num_warps=4,
        enable_fp_fusion=True,
    )
    if cache_values:
        value_rows = value_fp8.shape[-1] if fp8_values else segment_tokens
        cache_current_value[(parts, heads, triton.cdiv(value_rows, 32) * triton.cdiv(dim, 32))](
            value,
            metadata,
            value_fp8,
            value_scale,
            valid,
            key.shape[0],
            capacity,
            segment_tokens,
            value_rows,
            scale_max,
            heads,
            dim,
            fp8_values,
            *value.stride(),
            num_warps=4,
            enable_fp_fusion=False,
        )


@_encode_sage_current.register_fake
def _encode_sage_current_fake(
    key,
    value,
    metadata,
    position,
    key_int8,
    key_scale,
    key_mean,
    value_fp8,
    value_scale,
    valid,
    key_valid,
    capacity,
    architecture,
    cache_values,
    physical_start,
    enabled_after,
):
    return None


@lru_cache(maxsize=5)
def _native_attention(architecture):
    if architecture in (80, 86):
        from sageattention import _qattn_sm80

        def native_fp16(query, key, value, output, query_scale, key_scale, value_scale, *options):
            # Ampere keeps V once in NHD FP16, without FP8 scales/permutation.
            return _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(query, key, value[:, : key.shape[1]], output, query_scale, key_scale, *options)

        return native_fp16
    if architecture == 90:
        from sageattention import _qattn_sm90

        return _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
    from sageattention import _qattn_sm89

    return _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf


def _quantize_query(query, architecture):
    queries, heads, dim = query.shape
    query = query.unsqueeze(0)
    query_int8 = torch.empty_like(query, dtype=torch.int8)
    if architecture == 120:
        from sageattention import _fused

        query_scale = query.new_empty((1, heads, triton.cdiv(queries, 128) * 4), dtype=torch.float32)
        _fused.quant_per_warp_int8_cuda(query, query_int8, query_scale, 128, 32, 0)
    else:
        from sageattention.triton.quant_per_thread import quant_query_per_thread_int8_kernel

        block = 64 if architecture == 90 else 128
        warp = 16 if architecture == 90 else 32
        scales = triton.cdiv(queries, block) * 32
        query_scale = query.new_empty((1, heads, scales), dtype=torch.float32)
        quant_query_per_thread_int8_kernel[(scales, heads, 1)](
            query,
            query_int8,
            query_scale,
            queries,
            query.stride(0),
            query.stride(2),
            query.stride(1),
            query_int8.stride(0),
            query_int8.stride(2),
            query_int8.stride(1),
            query_scale.stride(0),
            query_scale.stride(1),
            C=dim,
            BLK=warp,
        )
    return query_int8, query_scale


def sage_kv_attention(query: torch.Tensor, state: SageKVCache, visible_tokens: int) -> torch.Tensor:
    return _sage_kv_attention(query, state.key_int8, state.key_scale, state.key_mean, state.value_fp8, state.value_scale, visible_tokens, state.architecture)


@torch.library.custom_op("lightx2v::sage_ring_attention", mutates_args=(), device_types="cuda")
def _sage_kv_attention(
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_mean: torch.Tensor,
    value_fp8: torch.Tensor,
    value_scale: torch.Tensor,
    visible_tokens: int,
    architecture: int,
) -> torch.Tensor:
    return _sage_kv_attention_impl(query, key_int8, key_scale, key_mean, value_fp8, value_scale, visible_tokens, architecture)


def _sage_kv_attention_impl(query, key_int8, key_scale, key_mean, value_fp8, value_scale, visible_tokens, architecture, *, return_lse=False):
    query = query.contiguous()
    queries, heads, dim = query.shape
    segment_tokens = key_int8.shape[1]
    full_parts, tail_tokens = divmod(visible_tokens, segment_tokens)
    parts = full_parts + bool(tail_tokens)
    query_int8, query_scale = _quantize_query(query, architecture)
    native = _native_attention(architecture)
    granularity = 2 if architecture == 120 else 3
    partial_output = query.new_empty((parts, queries, heads, dim))
    full_lse = query.new_empty((0,), dtype=torch.float32)
    tail_lse = full_lse
    if full_parts:
        # Sage CUDA kernels use 32-bit batch/row offsets. A large query and
        # many small KV segments can exceed that range in partial_output.
        batch_parts = max(1, (2**31 - 1) // (queries * heads * dim))
        lse_batches = []
        for start in range(0, full_parts, batch_parts):
            stop = min(start + batch_parts, full_parts)
            lse_batches.append(
                native(
                    query_int8.expand(stop - start, -1, -1, -1),
                    key_int8[start:stop],
                    value_fp8[start:stop],
                    partial_output[start:stop],
                    query_scale.expand(stop - start, -1, -1).contiguous(),
                    key_scale[start:stop],
                    value_scale[start:stop],
                    0,
                    0,
                    granularity,
                    dim**-0.5,
                    1,
                )
            )
        full_lse = lse_batches[0] if len(lse_batches) == 1 else torch.cat(lse_batches)
    if tail_tokens:
        key_block, per_thread, _ = _ARCH_CONFIG[architecture]
        tail_scales = triton.cdiv(tail_tokens, key_block) * (4 if per_thread else 1)
        tail_lse = native(
            query_int8,
            key_int8[full_parts : full_parts + 1, :tail_tokens],
            value_fp8[full_parts : full_parts + 1],
            partial_output[full_parts : full_parts + 1],
            query_scale,
            key_scale[full_parts : full_parts + 1, :, :tail_scales].contiguous(),
            value_scale[full_parts : full_parts + 1],
            0,
            0,
            granularity,
            dim**-0.5,
            1,
        )
    output = torch.empty_like(query)
    combined_lse = query.new_empty((queries, heads), dtype=torch.float32) if return_lse else None
    merge_attention[(triton.cdiv(queries, 4), heads)](
        partial_output,
        full_lse,
        tail_lse,
        query,
        key_mean,
        output,
        queries,
        heads,
        dim,
        full_parts,
        bool(tail_tokens),
        combined_lse=combined_lse,
        return_lse=return_lse,
        num_warps=4,
    )
    if return_lse:
        return output.flatten(1), combined_lse
    return output.flatten(1)


@_sage_kv_attention.register_fake
def _sage_kv_attention_fake(query, key_int8, key_scale, key_mean, value_fp8, value_scale, visible_tokens, architecture):
    return query.new_empty((query.shape[0], query.shape[1] * query.shape[2]))


def sage_kv_attention_with_lse(query: torch.Tensor, state: SageKVCache, visible_tokens: int):
    """Segmented output and natural-log LSE, including K-centering correction."""
    return _sage_kv_attention_with_lse(query, state.key_int8, state.key_scale, state.key_mean, state.value_fp8, state.value_scale, visible_tokens, state.architecture)


@torch.library.custom_op("lightx2v::sage_ring_attention_with_lse", mutates_args=(), device_types="cuda")
def _sage_kv_attention_with_lse(
    query: torch.Tensor,
    key_int8: torch.Tensor,
    key_scale: torch.Tensor,
    key_mean: torch.Tensor,
    value_fp8: torch.Tensor,
    value_scale: torch.Tensor,
    visible_tokens: int,
    architecture: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _sage_kv_attention_impl(query, key_int8, key_scale, key_mean, value_fp8, value_scale, visible_tokens, architecture, return_lse=True)


@_sage_kv_attention_with_lse.register_fake
def _sage_kv_attention_with_lse_fake(query, key_int8, key_scale, key_mean, value_fp8, value_scale, visible_tokens, architecture):
    return query.new_empty((query.shape[0], query.shape[1] * query.shape[2])), query.new_empty(query.shape[:2], dtype=torch.float32)


def sage_kv_attention_spans(query, states):
    """Merge disjoint quantized cache banks without constructing dense KV."""
    states = tuple((state, visible) for state, visible in states if visible > 0)
    if not states:
        raise ValueError("Segmented Sage attention requires a nonempty KV bank")
    if len(states) == 1:
        return sage_kv_attention(query, *states[0])
    outputs, scores = [], []
    for state, visible in states:
        output, lse = sage_kv_attention_with_lse(query, state, visible)
        outputs.append(output.view_as(query))
        scores.append(lse)
    weights = torch.softmax(torch.stack(scores).float(), dim=0)
    values = torch.stack(outputs)
    return (values.float() * weights.unsqueeze(-1)).sum(0).to(values.dtype).flatten(1)


class SageRingAttentionState(NamedTuple):
    key: torch.Tensor
    value: torch.Tensor
    slot_metadata: torch.Tensor
    visible_tokens: int
    quantized: SageKVCache
    logical_start: int = -1

    @property
    def bindings(self):
        return self.key, self.value, self.slot_metadata, *self.quantized.bindings


class QuantizedRingAttentionState(NamedTuple):
    slot_metadata: torch.Tensor
    visible_tokens: int
    quantized: SageKVCache
    capacity: int
    segments: tuple
    write_views: tuple

    @property
    def bindings(self):
        return self.slot_metadata, *self.quantized.bindings

    @property
    def graph_key(self):
        return self.capacity, self.quantized.key_int8.shape[1], self.segments


class SageRingAttention(RingAttention):
    """Quantize written ring segments and merge their attention outputs."""

    def __init__(self, allow_offload_history_reuse=False, *, profile="wan", segment_candidates=None):
        super().__init__()
        self.profile = profile
        if segment_candidates is not None:
            from .sage_tuning import validate_segment_candidates

            segment_candidates = validate_segment_candidates(segment_candidates)
        self.segment_candidates = segment_candidates
        self.segment_layouts = {}
        self.quantized_layers = {}
        self.allow_offload_history_reuse = allow_offload_history_reuse
        self.offload_history = {}
        self._active_quantized_layouts = {}

    def select_segment_tokens(self, key, **options):
        from .sage_policy import select_segment_tokens

        selected = select_segment_tokens(key, profile=self.profile, **options)
        if self.segment_candidates is not None:
            from .sage_tuning import select_sage_segment_tokens

            selected = select_sage_segment_tokens(key, self.segment_candidates, baseline=selected, **options).segment_tokens
        policy = tuple((name, value is not None if name == "freqs" else value) for name, value in options.items())
        self.segment_layouts[(tuple(key.shape), key.device, policy)] = selected
        return selected

    def prepare(self, pool, layer_id, update):
        if getattr(pool, "quantized_only", False):
            return self._prepare_quantized_only(pool, layer_id, update)
        pool._ensure_kv_buffer()
        capacity = pool.get_max_cache_shape(layer_id)
        write_tokens = update.write_end - update.write_start
        layout = replace(pool.ring_layout(layer_id))
        layout.roll(update.sink_tokens, update.evicted_tokens, pool.get_valid_length(layer_id))
        logical_start = update.attention_start if update.attention_start else -1
        if logical_start < 0 and layout.active and update.write_end < capacity:
            logical_start = 0
        visible_tokens = update.write_end - update.attention_start
        offload = getattr(pool, "_offload", None)
        preserve_raw_history = self.allow_offload_history_reuse and offload is not None and hasattr(pool, "prepare_quantized_attention") and logical_start == -1 and update.sink_tokens == 0
        write_alignment = (
            (update.write_start, update.write_end, *(offset for span in layout.spans(update.write_start, update.write_end) for offset in (span.physical_start, span.length)))
            if preserve_raw_history
            else ()
        )
        segment_tokens = self.select_segment_tokens(
            pool._key_buffer(layer_id), write_tokens=write_tokens, sink_tokens=update.sink_tokens, visible_tokens=visible_tokens, logical_start=logical_start, write_alignment=write_alignment
        )
        cache_key = id(pool), pool._metadata_index(layer_id), segment_tokens, logical_start
        quantized = self.get_quantized_cache(cache_key, pool._key_buffer(layer_id), segment_tokens)
        layer_key = cache_key[:2]
        if self._active_quantized_layouts.get(layer_key) != cache_key:
            quantized.valid.zero_()
            quantized.key_valid.zero_()
            self.offload_history.pop(cache_key, None)
            self._active_quantized_layouts[layer_key] = cache_key
        can_plan = (
            self.allow_offload_history_reuse
            and offload is not None
            and hasattr(pool, "prepare_quantized_attention")
            and logical_start == -1
            and update.sink_tokens == 0
            and write_tokens > 0
            and update.write_start % segment_tokens == 0
            and update.write_end % segment_tokens == 0
            and all(span.physical_start % segment_tokens == 0 and span.length % segment_tokens == 0 for span in layout.spans(update.write_start, update.write_end))
        )
        if can_plan:
            storage_token = offload.storage_generation, *(tensor.data_ptr() for tensor in quantized.bindings)
            history = self.offload_history.get(cache_key)
            if history is None or history["storage_token"] != storage_token:
                quantized.valid.zero_()
                quantized.key_valid.zero_()
                history = {"storage_token": storage_token, "key_parts": set(), "value_parts": set(), "initialized": False}
                self.offload_history[cache_key] = history
            written_parts = {
                part for span in layout.spans(update.write_start, update.write_end) for part in range(span.physical_start // segment_tokens, (span.physical_start + span.length) // segment_tokens)
            }
            visible_parts = set(range(triton.cdiv(visible_tokens, segment_tokens)))
            required_history = visible_parts - written_parts
            skip_raw_history = history["initialized"] and required_history <= history["key_parts"] and required_history <= history["value_parts"]

            def mark_initialized():
                # finish_attention calls this only after real kernel writes and
                # their raw CPU D2H copies have been queued. Subsequent staging
                # claims wait for the layer's write_done event before reuse.
                if offload.storage_generation == storage_token[0]:
                    history["key_parts"].update(visible_parts)
                    history["value_parts"].update(visible_parts)
                    history["initialized"] = True

            view = pool.prepare_quantized_attention(layer_id, update, skip_raw_history=skip_raw_history, on_finish=mark_initialized)
        else:
            if self.allow_offload_history_reuse and offload is not None:
                # A partial or logical-window update may use another quantized
                # view of the same raw layer. Do not reuse older full-ring
                # validity when returning to the aligned physical path.
                for other_key, history in self.offload_history.items():
                    if other_key[:2] == cache_key[:2]:
                        history["initialized"] = False
                        history["key_parts"].clear()
                        history["value_parts"].clear()
                        _, other_state = self.quantized_layers[other_key]
                        other_state.valid.zero_()
                        other_state.key_valid.zero_()
            view = pool.prepare_attention(layer_id, update)
        return SageRingAttentionState(view.key, view.value, view.slot_metadata, visible_tokens, quantized, logical_start)

    def _prepare_quantized_only(self, pool, layer_id, update):
        raw = pool.prepare_attention(layer_id, update)
        capacity = pool.get_max_cache_shape(layer_id)
        layout = pool._attention_layouts[pool._metadata_index(layer_id)]
        segment_tokens = self.select_segment_tokens(
            raw.key,
            capacity=capacity,
            write_tokens=update.write_end - update.write_start,
            sink_tokens=update.sink_tokens,
            value_only=True,
            write_alignment=pool.value_write_alignment,
        )
        boundaries = (capacity, update.write_start, update.write_end, update.attention_start, update.sink_tokens)
        if any(boundary % segment_tokens for boundary in boundaries):
            raise ValueError("Encoded-only Sage requires aligned ring write/window boundaries")
        cache_key = id(pool), pool._metadata_index(layer_id), "encoded"
        previous = self.quantized_layers.get(cache_key)
        if previous is not None and previous[1].key_int8.shape[1] != segment_tokens and update.write_start:
            raise ValueError("Encoded KV segment geometry changed while historical data was live; declare all write/window alignments before allocation")
        quantized = self.get_quantized_cache(cache_key, raw.key, segment_tokens, capacity=capacity)
        spans = tuple(layout.spans(update.attention_start, update.write_end))
        segments = tuple((span.physical_start // segment_tokens, span.length // segment_tokens) for span in spans)
        if (update.attention_start == 0 and update.write_end == capacity) or (len(spans) == 1 and spans[0].physical_start == 0):
            segments = ()
        dirty = {part for span in layout.spans(update.write_start, update.write_end) for part in range(span.physical_start // segment_tokens, (span.physical_start + span.length) // segment_tokens)}
        # MG4 may rotate its small sink when the relative anchor moves. Including
        # this prefix also makes capture snapshots safe for that model's hook.
        dirty.update(range(update.sink_tokens // segment_tokens))
        write_views = tuple(tensor[part : part + 1] for part in sorted(dirty) for tensor in quantized.bindings if tensor.numel())
        return QuantizedRingAttentionState(raw.slot_metadata, update.write_end - update.attention_start, quantized, capacity, segments, write_views)

    def get_quantized_cache(self, cache_key, key, segment_tokens, *, capacity=None):
        signature = key.data_ptr(), key.shape, key.stride(), key.dtype, key.device, segment_tokens, capacity
        cached = self.quantized_layers.get(cache_key)
        if cached is None or cached[0] != signature:
            cached = signature, allocate_sage_kv_cache(key, segment_tokens, capacity=capacity)
            self.quantized_layers[cache_key] = cached
        return cached[1]

    def update(self, key, value, state):
        if isinstance(state, QuantizedRingAttentionState):
            encode_sage_current(key, value, state.slot_metadata, state.quantized, state.capacity)
            return
        from lightx2v.common.kvcache.kernel import store_kv_ring

        store_kv_ring(key, value, state.key, state.value, state.slot_metadata)
        update_sage_kv_cache(state.key, state.value, state.slot_metadata, state.quantized, key.shape[0], state.visible_tokens, logical_start=state.logical_start)

    def apply(self, query, key, value, state, attention):
        self.update(key, value, state)
        if isinstance(state, QuantizedRingAttentionState) and state.segments:
            encoded, segment_tokens = state.quantized, state.quantized.key_int8.shape[1]
            spans = tuple((SageKVCache(*(tensor[start : start + parts] for tensor in encoded.bindings), encoded.architecture), parts * segment_tokens) for start, parts in state.segments)
            return sage_kv_attention_spans(query, spans)
        return sage_kv_attention(query, state.quantized, state.visible_tokens)

    def reset(self, keep_storage=False):
        super().reset(keep_storage)
        self.offload_history.clear()
        self._active_quantized_layouts.clear()
        if keep_storage:
            for _, state in self.quantized_layers.values():
                state.valid.zero_()
                state.key_valid.zero_()
        else:
            self.quantized_layers.clear()
