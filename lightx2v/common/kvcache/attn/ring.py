"""Attention over physical ring views, with a bounded workspace for narrow windows."""

from typing import NamedTuple

import torch

from lightx2v.common.kvcache.base import KVCacheView
from lightx2v.common.kvcache.rolling import RingKVCacheView


class WindowRingAttentionView(NamedTuple):
    raw: RingKVCacheView
    indices: torch.Tensor
    workspace: KVCacheView

    @property
    def bindings(self):
        return (*self.raw.bindings, self.indices, *self.workspace.bindings)


class RingAttention:
    def __init__(self):
        self.window_workspaces = {}

    def prepare(self, pool, layer_id, update):
        raw = pool.prepare_attention(layer_id, update)
        if update.attention_start == 0:
            return raw
        length = update.write_end - update.attention_start
        cache_key = id(pool), pool._metadata_index(layer_id), length
        cached = self.window_workspaces.get(cache_key)
        if cached is None or cached.workspace.key.shape[1:] != raw.key.shape[1:]:
            shape = (length, *raw.key.shape[1:])
            cached = WindowRingAttentionView(raw, torch.empty(length, dtype=torch.int64, device=raw.key.device), KVCacheView(raw.key.new_empty(shape), raw.value.new_empty(shape)))
            self.window_workspaces[cache_key] = cached
        indices = [index for span in pool.ring_layout(layer_id).spans(update.attention_start, update.write_end) for index in range(span.physical_start, span.physical_start + span.length)]
        cached.indices.copy_(torch.tensor(indices, dtype=torch.int64), non_blocking=True)
        return WindowRingAttentionView(raw, cached.indices, cached.workspace)

    @staticmethod
    def update(key, value, state):
        from lightx2v.common.kvcache.kernel import store_kv_ring

        raw = state.raw if isinstance(state, WindowRingAttentionView) else state
        store_kv_ring(key, value, raw.key, raw.value, raw.slot_metadata)
        if isinstance(state, WindowRingAttentionView):
            torch.index_select(raw.key, 0, state.indices, out=state.workspace.key)
            torch.index_select(raw.value, 0, state.indices, out=state.workspace.value)
            return state.workspace
        return KVCacheView(raw.key[: raw.visible_tokens], raw.value[: raw.visible_tokens])

    def apply(self, query, key, value, state, attention):
        physical = self.update(key, value, state)
        return attention.apply(q=query, k=physical.key, v=physical.value)

    def reset(self, keep_storage=False):
        if not keep_storage:
            self.window_workspaces.clear()
