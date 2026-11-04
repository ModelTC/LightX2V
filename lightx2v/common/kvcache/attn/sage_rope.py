"""Raw K plus persistent native-dtype V, with original-position RoPE."""

from typing import NamedTuple

import torch
import triton

from ..layout import KVCacheUpdate
from ..rolling import RingKVCacheView, StepRollingKVCachePool
from .rope_utils import RotaryFrequencyCache, apply_model_rope
from .sage import SageKVCache, SageRingAttention, sage_kv_attention, sage_kv_attention_spans, update_sage_kv_cache_from_current


@torch.library.custom_op("lightx2v::sage_ring_rope_query", mutates_args=(), device_types="cuda")
def _rotate_query(query: torch.Tensor, freqs: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
    from .kernels.sage_ring import rotate_query

    query = query.contiguous()
    output = torch.empty_like(query)
    rotate_query[(triton.cdiv(query.numel(), 512),)](query, output, freqs, metadata, query.numel(), query.shape[1], query.shape[2], 512)
    return output


@_rotate_query.register_fake
def _rotate_query_fake(query, freqs, metadata):
    return torch.empty_like(query, memory_format=torch.contiguous_format)


class SageRoPERingView(NamedTuple):
    raw: RingKVCacheView
    quantized: SageKVCache
    freqs: torch.Tensor
    logical_start: int = -1
    rotary_dim: int = 0
    rope_layout: str = "interleaved"
    rope_fma: bool = True
    segments: tuple = ()
    physical_tokens: int = 0
    refresh_keys: bool = False

    @property
    def bindings(self):
        return (*self.raw.bindings, *self.quantized.bindings, self.freqs)

    @property
    def graph_key(self):
        return self.logical_start, self.rotary_dim, self.rope_layout, self.rope_fma, self.segments, self.physical_tokens, self.refresh_keys


class SageRoPERingAttention(SageRingAttention):
    """Rotate raw K, persist native V and share temporary K for step banks."""

    def __init__(self, *, rotary_dim=None, rope_layout="interleaved", rope_fma=True, profile="wan", share_step_keys=False):
        super().__init__(profile=profile)
        if rope_layout not in ("interleaved", "split_half"):
            raise ValueError("RoPE layout must be interleaved or split_half")
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self.rope_fma = rope_fma
        self.share_step_keys = share_step_keys
        self._step_key_workspaces = {}
        self._generations = {}
        self._pool_segments = {}
        self._write_views = {}
        self._frequencies = RotaryFrequencyCache()

    def prepare(self, pool, layer_id, update, *, freqs):
        """Audio/step-cache entry, with a fresh anchor for each output chunk."""
        self._check_pool(pool)
        raw = pool.prepare_attention(layer_id, update)
        cache_key = id(pool), pool._metadata_index(layer_id)
        return self._prepare(pool, layer_id, update, raw, cache_key, freqs, update.global_end)

    @staticmethod
    def _check_pool(pool):
        if not pool.key_only:
            pool.configure_key_only()
        pool._ensure_kv_buffer()

    def prepare_rope_ring(self, pool, layer_id, *, cache_key, current_tokens, visible_tokens, sink_tokens, freqs, generation):
        self._check_pool(pool)
        end = pool.get_valid_length(layer_id)
        evicted = max(0, end + current_tokens - visible_tokens)
        update = KVCacheUpdate(end - evicted, end - evicted + current_tokens, 0, 0, sink_tokens, evicted)
        raw = pool.prepare_pending_attention(layer_id, update)
        return self._prepare(pool, layer_id, update, raw, cache_key, freqs, generation)

    def _prepare(self, pool, layer_id, update, raw, cache_key, freqs, generation):
        rotary_dim = self.rotary_dim or raw.key.shape[-1]
        prepared = self._frequencies.prepare(freqs, rotary_dim, self.rope_layout)
        layout = pool._attention_layouts[pool._metadata_index(layer_id)]
        capacity = raw.key.shape[0]
        partial = update.attention_start > 0 or (layout.active and update.write_end < capacity)
        spans = layout.spans(update.attention_start, update.write_end) if partial else ()
        layer_key = id(pool), pool._metadata_index(layer_id)
        segment_tokens = self._pool_segments.get(layer_key)
        alignment = (*pool.value_write_alignment, update.write_start, update.write_end, update.attention_start, *(offset for span in spans for offset in (span.physical_start, span.length)))
        if segment_tokens is None:
            segment_tokens = self.select_segment_tokens(
                raw.key,
                write_tokens=update.write_end - update.write_start,
                sink_tokens=update.sink_tokens,
                value_only=True,
                write_alignment=alignment,
            )
            self._pool_segments[layer_key] = segment_tokens
        if any(offset % segment_tokens for offset in alignment):
            raise ValueError("Quantized V writes and visible spans must use the pool's fixed segment alignment")
        quant_key = id(pool), pool._metadata_index(layer_id), segment_tokens
        quantized = self.get_quantized_cache(quant_key, raw.key, segment_tokens)
        shared_keys = self.share_step_keys and isinstance(pool, StepRollingKVCachePool)
        if shared_keys:
            # Each step bank retains its raw K and encoded V. Rotated INT8 K
            # is rebuilt for every use, so sequential layers and steps share its
            # workspace without changing their histories or RoPE positions.
            workspace_key = id(pool), tuple(raw.key.shape), raw.key.dtype, raw.key.device, segment_tokens
            keys = self._step_key_workspaces.setdefault(workspace_key, (quantized.key_int8, quantized.key_scale, quantized.key_mean))
            if quantized.key_int8 is not keys[0]:
                quantized = quantized._replace(key_int8=keys[0], key_scale=keys[1], key_mean=keys[2])
                signature, _ = self.quantized_layers[quant_key]
                self.quantized_layers[quant_key] = signature, quantized
        generation = generation, self._frequencies.last_revision
        if not shared_keys and self._generations.get(quant_key) != generation:
            quantized.key_valid.zero_()
            self._generations[quant_key] = generation
        self._write_views[cache_key] = pool.write_views(layer_id, update.write_start, update.write_end)
        segments = tuple((span.physical_start // segment_tokens, span.length // segment_tokens) for span in spans)
        return SageRoPERingView(raw, quantized, prepared, -1, rotary_dim, self.rope_layout, self.rope_fma, segments, capacity if partial else update.write_end, shared_keys)

    def write_views(self, cache_key):
        return self._write_views[cache_key]

    @staticmethod
    def update_rope_ring(key, value, cache):
        from ..kernel import store_key_ring

        if cache.refresh_keys:
            # Keep this inside captured computation: graph warmup/replay can
            # use the workspace for other layers or step banks in between.
            cache.quantized.key_valid.zero_()
        store_key_ring(key, cache.raw.key, cache.raw.slot_metadata)
        update_sage_kv_cache_from_current(
            cache.raw.key,
            value,
            cache.raw.slot_metadata,
            cache.quantized,
            key.shape[0],
            cache.raw.visible_tokens,
            freqs=cache.freqs,
            logical_start=cache.logical_start,
            rotary_dim=cache.rotary_dim or cache.raw.key.shape[-1],
            rope_layout=cache.rope_layout,
            rope_fma=cache.rope_fma,
            physical_tokens=cache.physical_tokens,
        )

    @staticmethod
    def _attend(query, cache):
        if not cache.segments:
            return sage_kv_attention(query, cache.quantized, cache.raw.visible_tokens)
        state = cache.quantized
        segment_tokens = state.key_int8.shape[1]
        spans = tuple((SageKVCache(*(tensor[start : start + parts] for tensor in state.bindings), state.architecture), parts * segment_tokens) for start, parts in cache.segments)
        return sage_kv_attention_spans(query, spans)

    def apply_rope_ring(self, query, key, value, cache, *, attention, rope, query_freqs):
        # Current KV must be visible to fused quantization in its physical slots.
        self.update_rope_ring(key, value, cache)
        query = apply_model_rope(rope, query, query_freqs, cache.rotary_dim or query.shape[-1])
        return self._attend(query, cache)

    def apply(self, query, key, value, cache, attention):
        self.update_rope_ring(key, value, cache)
        if cache.rope_layout != "interleaved" or cache.rotary_dim not in (0, query.shape[-1]) or not cache.rope_fma:
            from ..kernel import rotate_token_rope

            query = rotate_token_rope(query, cache.freqs, cache.raw.slot_metadata, rotary_dim=cache.rotary_dim, split_half=cache.rope_layout == "split_half", rope_fma=cache.rope_fma)
        else:
            query = _rotate_query(query, cache.freqs, cache.raw.slot_metadata)
        return self._attend(query, cache)

    def reset(self, keep_storage=False):
        super().reset(keep_storage)
        self._generations.clear()
        self._pool_segments.clear()
        self._write_views.clear()
        if not keep_storage:
            self._step_key_workspaces.clear()
            self._frequencies.clear()
