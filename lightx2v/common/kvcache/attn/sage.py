"""Persistent, incrementally quantized physical KV segments for SageAttention2."""

from functools import lru_cache
from math import gcd
from typing import NamedTuple

import torch
import triton

from .kernels.sage_ring import mark_valid, merge_attention, quantize_key, quantize_value, segment_stats
from .ring import RingAttention, WindowRingAttentionView

# Native SageAttention2 dispatch: Ada, Hopper, and Blackwell consumer GPUs.
_ARCH_CONFIG = {89: (64, True, 2.25), 90: (128, True, 448.0), 120: (64, False, 2.25)}


class SageKVCache(NamedTuple):
    key_int8: torch.Tensor
    key_scale: torch.Tensor
    key_mean: torch.Tensor
    value_fp8: torch.Tensor
    value_scale: torch.Tensor
    valid: torch.Tensor
    architecture: int

    @property
    def bindings(self) -> tuple[torch.Tensor, ...]:
        return self.key_int8, self.key_scale, self.key_mean, self.value_fp8, self.value_scale, self.valid


def allocate_sage_kv_cache(key_ring: torch.Tensor, segment_tokens: int) -> SageKVCache:
    """Allocate a quantized companion of a resident [tokens, heads, dim] KV ring."""
    major, minor = torch.cuda.get_device_capability(key_ring.device)
    architecture = major * 10 + minor
    if architecture not in _ARCH_CONFIG:
        raise ValueError("SageAttention2 ring caching requires SM89, SM90, or SM120")
    capacity, heads, dim = key_ring.shape
    if dim not in (64, 128) or not 0 < segment_tokens <= capacity:
        raise ValueError("SageAttention2 ring caching requires head_dim 64/128 and a nonempty segment within capacity")
    key_block, per_thread, _ = _ARCH_CONFIG[architecture]
    parts = triton.cdiv(capacity, segment_tokens)
    scales = triton.cdiv(segment_tokens, key_block) * (4 if per_thread else 1)
    padded_tokens = triton.cdiv(segment_tokens, 128 if architecture == 90 else 64) * (128 if architecture == 90 else 64)
    return SageKVCache(
        key_ring.new_empty((parts, segment_tokens, heads, dim), dtype=torch.int8),
        key_ring.new_empty((parts, heads, scales), dtype=torch.float32),
        key_ring.new_empty((parts, heads, dim)),
        key_ring.new_empty((parts, dim, heads, padded_tokens), dtype=torch.float8_e4m3fn),
        key_ring.new_empty((parts, heads, dim), dtype=torch.float32),
        torch.zeros(parts, device=key_ring.device, dtype=torch.int32),
        architecture,
    )


def update_sage_kv_cache(key_ring, value_ring, metadata, state: SageKVCache, write_tokens: int, visible_tokens: int) -> None:
    """Requantize initialized or newly written segments after the ring slot write."""
    _update_sage_kv_cache(key_ring, value_ring, metadata, *state.bindings, write_tokens, visible_tokens, state.architecture)


@torch.library.custom_op(
    "lightx2v::update_sage_kv_cache",
    mutates_args=("key_int8", "key_scale", "key_mean", "value_fp8", "value_scale", "valid"),
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
    write_tokens: int,
    visible_tokens: int,
    architecture: int,
) -> None:
    key_block, per_thread, scale_max = _ARCH_CONFIG[architecture]
    _, segment_tokens, heads, dim = key_int8.shape
    parts = triton.cdiv(visible_tokens, segment_tokens)
    capacity = key_ring.shape[0]
    segment_stats[(parts, heads, triton.cdiv(dim, 32))](
        key_ring,
        value_ring,
        metadata,
        valid,
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
        num_warps=4,
    )
    quantize_key[(parts, heads, key_scale.shape[-1])](
        key_ring,
        metadata,
        valid,
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
        num_warps=4,
    )
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
        num_warps=4,
    )
    mark_valid[(1,)](valid, parts, triton.next_power_of_2(parts))


@_update_sage_kv_cache.register_fake
def _update_sage_kv_cache_fake(key_ring, value_ring, metadata, key_int8, key_scale, key_mean, value_fp8, value_scale, valid, write_tokens, visible_tokens, architecture):
    return None


@lru_cache(maxsize=3)
def _native_attention(architecture):
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
        full_lse = native(
            query_int8.expand(full_parts, -1, -1, -1),
            key_int8[:full_parts],
            value_fp8[:full_parts],
            partial_output[:full_parts],
            query_scale.expand(full_parts, -1, -1).contiguous(),
            key_scale[:full_parts],
            value_scale[:full_parts],
            0,
            0,
            granularity,
            dim**-0.5,
            1,
        )
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
    merge_attention[(queries, heads)](partial_output, full_lse, tail_lse, query, key_mean, output, queries, heads, dim, full_parts, bool(tail_tokens), num_warps=1)
    return output.flatten(1)


@_sage_kv_attention.register_fake
def _sage_kv_attention_fake(query, key_int8, key_scale, key_mean, value_fp8, value_scale, visible_tokens, architecture):
    return query.new_empty((query.shape[0], query.shape[1] * query.shape[2]))


class SageRingAttentionState(NamedTuple):
    key: torch.Tensor
    value: torch.Tensor
    slot_metadata: torch.Tensor
    visible_tokens: int
    quantized: SageKVCache

    @property
    def bindings(self):
        return self.key, self.value, self.slot_metadata, *self.quantized.bindings


class SageRingAttention(RingAttention):
    """Quantize written ring segments and merge their attention outputs."""

    def __init__(self):
        super().__init__()
        self.quantized_layers = {}

    def prepare(self, pool, layer_id, update):
        view = super().prepare(pool, layer_id, update)
        if isinstance(view, WindowRingAttentionView):
            return view
        write_tokens = update.write_end - update.write_start
        segment_tokens = gcd(view.key.shape[0], write_tokens, update.sink_tokens)
        quantized = self.get_quantized_cache((id(pool), pool._metadata_index(layer_id)), view.key, segment_tokens)
        return SageRingAttentionState(*view, quantized)

    def get_quantized_cache(self, cache_key, key, segment_tokens):
        signature = key.data_ptr(), key.shape, key.stride(), key.dtype, key.device
        cached = self.quantized_layers.get(cache_key)
        if cached is None or cached[0] != signature:
            cached = signature, allocate_sage_kv_cache(key, segment_tokens)
            self.quantized_layers[cache_key] = cached
        return cached[1]

    def update(self, key, value, state):
        physical = super().update(key, value, state)
        if isinstance(state, WindowRingAttentionView):
            return physical
        update_sage_kv_cache(state.key, state.value, state.slot_metadata, state.quantized, key.shape[0], state.visible_tokens)
        return physical

    def apply(self, query, key, value, state, attention):
        if isinstance(state, WindowRingAttentionView):
            return super().apply(query, key, value, state, attention)
        self.update(key, value, state)
        return sage_kv_attention(query, state.quantized, state.visible_tokens)

    def reset(self, keep_storage=False):
        super().reset(keep_storage)
        if keep_storage:
            for _, state in self.quantized_layers.values():
                state.valid.zero_()
        else:
            self.quantized_layers.clear()
