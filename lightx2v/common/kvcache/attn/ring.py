"""Attention over physical ring views with workspaces for bounded windows."""

from typing import NamedTuple

from lightx2v.common.kvcache.base import KVCacheView
from lightx2v.common.kvcache.rolling import RingKVCacheView

from .flash import FlashAttentionWorkspace, apply_flash_attention


class WindowRingAttentionView(NamedTuple):
    raw: RingKVCacheView
    segments: tuple
    workspace: tuple

    @property
    def key(self):
        return self.raw.key

    @property
    def bindings(self):
        spans = tuple(self.raw.value[physical : physical + length] for _, physical, length in self.segments)
        return (*self.raw.bindings, *self.workspace, *spans)

    @property
    def graph_key(self):
        return self.segments


class RingAttention:
    def __init__(self):
        self.flash_workspace = FlashAttentionWorkspace()

    def prepare(self, pool, layer_id, update):
        raw = pool.prepare_attention(layer_id, update)
        layout = pool._attention_layouts[pool._metadata_index(layer_id)]
        if update.attention_start == 0 and (not layout.active or update.write_end == raw.key.shape[0]):
            return raw
        segments = tuple(tuple(span) for span in layout.spans(update.attention_start, update.write_end))
        workspace = self.flash_workspace.prepare(id(pool), raw.key, update.write_end - update.attention_start) if len(segments) > 1 else ()
        return WindowRingAttentionView(raw, segments, workspace)

    @staticmethod
    def update(key, value, state):
        from lightx2v.common.kvcache.kernel import store_kv_ring

        raw = state.raw if isinstance(state, WindowRingAttentionView) else state
        store_kv_ring(key, value, raw.key, raw.value, raw.slot_metadata)
        return KVCacheView(raw.key[: raw.visible_tokens], raw.value[: raw.visible_tokens])

    def apply(self, query, key, value, state, attention):
        physical = self.update(key, value, state)
        if isinstance(state, WindowRingAttentionView):
            spans = tuple((state.raw.key[start : start + length], state.raw.value[start : start + length]) for _, start, length in state.segments)
            return apply_flash_attention(query, spans, attention, workspace=state.workspace)
        return attention.apply(q=query, k=physical.key, v=physical.value)

    def reset(self, keep_storage=False):
        self.flash_workspace.reset(keep_storage)
