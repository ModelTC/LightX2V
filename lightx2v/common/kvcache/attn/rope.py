"""Reusable logical KV workspace for post-RoPE attention."""

from typing import NamedTuple

import torch

from ..base import KVCacheView
from ..layout import KVCacheUpdate
from ..rolling import RingKVCacheView
from ..rolling_offload import OffloadedRollingKVCachePool
from .ring import RingAttention


class RoPEKVWorkspace:
    def __init__(self):
        self.history = {}
        self._span_metadata = {}

    def prepare_history(
        self,
        pool,
        layer_id: int,
        *,
        cache_key,
        generation: int,
        capacity: int,
        visible_tokens: int,
        ranges: tuple[tuple[int, int], ...],
        freqs: torch.Tensor,
        prototype: torch.Tensor,
    ) -> KVCacheView:
        from ..kernel import gather_kv_cache_rope

        pool._ensure_kv_buffer()
        key_cache, value_cache = pool._key_buffer(layer_id), pool._value_buffer(layer_id)
        cached = self.history.get(cache_key)
        shape = (capacity, *key_cache.shape[1:])
        if cached is None or cached[1].shape != shape:
            key = prototype.new_empty(shape)
            cached = -1, key, torch.empty_like(key)
        previous_generation, key_storage, value_storage = cached
        key, value = key_storage[:visible_tokens], value_storage[:visible_tokens]
        if previous_generation != generation:
            spans = []
            output_start = 0
            for start, end in ranges:
                for span in pool.ring_layout(layer_id).spans(start, end):
                    spans.append((span.physical_start, output_start + span.logical_start - start, span.length))
                output_start += end - start
            if output_start:
                span_key = key_cache.device, tuple(spans)
                metadata = self._span_metadata.get(span_key)
                if metadata is None:
                    metadata = torch.tensor(spans, dtype=torch.int32, device=key_cache.device)
                    self._span_metadata[span_key] = metadata
                gather_kv_cache_rope(key_cache, value_cache, key[:output_start], value[:output_start], freqs[:output_start], metadata)
            self.history[cache_key] = generation, key_storage, value_storage
        return KVCacheView(key, value)

    @staticmethod
    def update_rope(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, cache: KVCacheView, freqs: torch.Tensor) -> KVCacheView:
        from ..kernel import update_kv_cache_rope

        current_tokens = key.shape[0]
        update_kv_cache_rope(query, key, value, cache.key[-current_tokens:], cache.value[-current_tokens:], freqs)
        return cache

    def reset(self, keep_storage: bool = False) -> None:
        if keep_storage:
            self.history = {cache_key: (-1, key, value) for cache_key, (_, key, value) in self.history.items()}
        else:
            self.history.clear()
            self._span_metadata.clear()


class RoPERingAttentionView(NamedTuple):
    raw: RingKVCacheView
    workspace: KVCacheView
    freqs: torch.Tensor

    @property
    def bindings(self) -> tuple[torch.Tensor, ...]:
        return (*self.raw.bindings, *self.workspace.bindings, self.freqs)


class RoPERingAttention(RingAttention):
    """Write raw ring slots and prepare a reusable ordered RoPE workspace."""

    def __init__(self):
        self.workspace = RoPEKVWorkspace()
        self._write_views = {}

    def prepare_rope_ring(self, pool, layer_id, *, cache_key, current_tokens, visible_tokens, sink_tokens, freqs, generation):
        end = pool.get_valid_length(layer_id)
        evicted = max(0, end + current_tokens - visible_tokens)
        update = KVCacheUpdate(end - evicted, end - evicted + current_tokens, 0, 0, sink_tokens, evicted)
        raw = pool.prepare_pending_attention(layer_id, update)
        ranges = ((0, sink_tokens), (sink_tokens + evicted, end)) if evicted else ((0, end),)
        workspace_key = cache_key
        if isinstance(pool, OffloadedRollingKVCachePool):
            workspace_key = (cache_key[0], "staging")
            generation = generation, layer_id
        workspace = self.workspace.prepare_history(
            pool, layer_id, cache_key=workspace_key, generation=generation, capacity=pool.get_max_cache_shape(layer_id), visible_tokens=visible_tokens, ranges=ranges, freqs=freqs, prototype=raw.key
        )
        self._write_views[cache_key] = (*pool.write_views(layer_id, update.write_start, update.write_end), workspace.key[-current_tokens:], workspace.value[-current_tokens:])
        return RoPERingAttentionView(raw, workspace, freqs[-current_tokens:])

    def write_views(self, cache_key):
        return self._write_views[cache_key]

    @staticmethod
    def update_rope_ring(key, value, cache):
        RingAttention.update(key, value, cache.raw)

    def apply_rope_ring(self, query, key, value, cache, *, attention, rope, query_freqs):
        self.update_rope_ring(key, value, cache)
        if query.shape[0] == key.shape[0]:
            self.workspace.update_rope(query, key, value, cache.workspace, cache.freqs)
        else:
            query = rope.apply_single(query, query_freqs)
            cache.workspace.key[-key.shape[0] :].copy_(rope.apply_single(key, cache.freqs))
            cache.workspace.value[-value.shape[0] :].copy_(value)
        return attention.apply(q=query, k=cache.workspace.key, v=cache.workspace.value)

    def reset(self, keep_storage=False):
        self.workspace.reset(keep_storage)
        self._write_views.clear()


class PhysicalRoPERingAttentionView(NamedTuple):
    raw: RingKVCacheView
    key: torch.Tensor
    freqs: torch.Tensor

    @property
    def bindings(self) -> tuple[torch.Tensor, ...]:
        return (*self.raw.bindings, self.key, self.freqs)


class PhysicalRoPERingAttention(RingAttention):
    """Rotate keys in physical slot order and read values directly from the ring."""

    def __init__(self):
        super().__init__()
        self.key_workspaces = {}

    def prepare(self, pool, layer_id, update, *, freqs):
        raw = super().prepare(pool, layer_id, update)
        capacity = pool.get_max_cache_shape(layer_id)
        key = self.key_workspaces.get(capacity)
        if key is None or key.shape != raw.key.shape:
            key = self.key_workspaces[capacity] = torch.empty_like(raw.key)
        return PhysicalRoPERingAttentionView(raw, key[: update.write_end], freqs)

    def apply(self, query, key, value, cache, attention):
        from ..kernel import rotate_ring_qk

        self.update(key, value, cache.raw)
        rotate_ring_qk(query, cache.raw.key, cache.key, cache.freqs, cache.raw.slot_metadata)
        return attention.apply(q=query, k=cache.key, v=cache.raw.value[: cache.raw.visible_tokens])

    def reset(self, keep_storage=False):
        if not keep_storage:
            self.key_workspaces.clear()
