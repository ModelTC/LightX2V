"""Post-RoPE FlashAttention over physical KV slots, with no V workspace."""

from typing import NamedTuple

import torch

from ..layout import KVCacheUpdate
from ..rolling import RingKVCacheView
from ..rolling_offload import OffloadedRollingKVCachePool
from .flash import apply_flash_attention
from .ring import RingAttention
from .rope_utils import RotaryFrequencyCache, apply_model_rope


class PhysicalRoPERingAttentionView(NamedTuple):
    raw: RingKVCacheView
    key: torch.Tensor
    freqs: torch.Tensor
    segments: tuple
    rotary_dim: int = 0
    rope_layout: str = "interleaved"
    rope_fma: bool = True
    workspace: tuple = ()

    @property
    def bindings(self) -> tuple[torch.Tensor, ...]:
        spans = tuple(self.raw.value[physical : physical + length] for _, physical, length in self.segments)
        return (*self.raw.bindings, self.key, self.freqs, *self.workspace, *spans)

    @property
    def graph_key(self):
        return self.segments


class PhysicalRoPERingAttention(RingAttention):
    """Rotate keys in physical slot order and read values directly from the ring."""

    def __init__(self, *, rotary_dim=None, rope_layout="interleaved", rope_fma=True):
        super().__init__()
        if rope_layout not in ("interleaved", "split_half"):
            raise ValueError("RoPE layout must be interleaved or split_half")
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self.rope_fma = rope_fma
        self._frequencies = RotaryFrequencyCache()
        self.key_workspaces = {}
        self._write_views = {}

    def prepare_rope_ring(self, pool, layer_id, *, cache_key, current_tokens, visible_tokens, sink_tokens, freqs, generation):
        end = pool.get_valid_length(layer_id)
        evicted = max(0, end + current_tokens - visible_tokens)
        update = KVCacheUpdate(end - evicted, end - evicted + current_tokens, 0, 0, sink_tokens, evicted)
        raw = pool.prepare_pending_attention(layer_id, update)
        return self._prepare_view(pool, layer_id, update, raw, freqs, cache_key)

    def prepare(self, pool, layer_id, update, *, freqs):
        """Shared committed-cache interface used by the audio AR runners."""
        raw = pool.prepare_attention(layer_id, update)
        cache_key = id(pool), pool._metadata_index(layer_id)
        return self._prepare_view(pool, layer_id, update, raw, freqs, cache_key)

    def _key_workspace_key(self, pool, cache_key, raw):
        return (cache_key[0], "staging") if isinstance(pool, OffloadedRollingKVCachePool) else cache_key

    def _prepare_view(self, pool, layer_id, update, raw, freqs, cache_key):
        capacity = pool.get_max_cache_shape(layer_id)
        visible_tokens = update.write_end
        layout = pool._attention_layouts[pool._metadata_index(layer_id)]
        workspace_key = self._key_workspace_key(pool, cache_key, raw)
        key = self.key_workspaces.get(workspace_key)
        if key is None or key.shape != raw.key.shape or key.device != raw.key.device or key.dtype != raw.key.dtype:
            key = self.key_workspaces[workspace_key] = torch.empty_like(raw.key)
        key = key[:visible_tokens] if visible_tokens == capacity or not layout.active else key
        if update.attention_start == 0 and (visible_tokens == capacity or not layout.active):
            segments = ((0, 0, visible_tokens),)
        else:
            segments = tuple(tuple(span) for span in layout.spans(update.attention_start, visible_tokens))
        workspace = self.flash_workspace.prepare(id(pool), raw.key, visible_tokens - update.attention_start) if len(segments) > 1 else ()
        self._write_views[cache_key] = pool.write_views(layer_id, update.write_start, update.write_end)
        rotary_dim = self.rotary_dim or raw.key.shape[-1]
        freqs = self._frequencies.prepare(freqs, rotary_dim, self.rope_layout)
        return PhysicalRoPERingAttentionView(raw, key, freqs, segments, rotary_dim, self.rope_layout, self.rope_fma, workspace)

    def write_views(self, cache_key):
        return self._write_views[cache_key]

    @staticmethod
    def update_rope_ring(key, value, cache):
        RingAttention.update(key, value, cache.raw)

    def apply_rope_ring(self, query, key, value, cache, *, attention, rope, query_freqs):
        from ..kernel import rotate_ring_qk, rotate_token_rope

        self.update_rope_ring(key, value, cache)
        rotary_dim = cache.rotary_dim or query.shape[-1]
        generic_rope = cache.rope_layout != "interleaved" or rotary_dim != query.shape[-1] or not cache.rope_fma
        if generic_rope:
            query = (
                rotate_token_rope(query, cache.freqs, cache.raw.slot_metadata, rotary_dim=rotary_dim, split_half=cache.rope_layout == "split_half", rope_fma=cache.rope_fma)
                if rope is None
                else apply_model_rope(rope, query, query_freqs, rotary_dim)
            )
            rotate_ring_qk(query[:0], cache.raw.key, cache.key, cache.freqs, cache.raw.slot_metadata, rotary_dim=rotary_dim, rope_layout=cache.rope_layout, rope_fma=cache.rope_fma)
        elif query.shape[0] == key.shape[0]:
            rotate_ring_qk(query, cache.raw.key, cache.key, cache.freqs, cache.raw.slot_metadata)
        else:
            query = rope.apply_single(query, query_freqs)
            # Empty Q prevents the key rotation kernel from rotating local Q a
            # second time when sequence parallelism splits its token range.
            rotate_ring_qk(query[:0], cache.raw.key, cache.key, cache.freqs, cache.raw.slot_metadata)
        spans = tuple((cache.key[start : start + length], cache.raw.value[start : start + length]) for _, start, length in cache.segments)
        return apply_flash_attention(query, spans, attention, workspace=cache.workspace)

    def apply(self, query, key, value, cache, attention):
        return self.apply_rope_ring(query, key, value, cache, attention=attention, rope=None, query_freqs=None)

    def reset(self, keep_storage=False):
        super().reset(keep_storage)
        if not keep_storage:
            self.key_workspaces.clear()
            self._frequencies.clear()
        self._write_views.clear()
