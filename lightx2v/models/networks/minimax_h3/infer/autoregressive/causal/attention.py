"""H3-specific raw-V cache for variable audio/video and condition row writes.

A common V-only segment would be one or two tokens for real H3 AV windows,
which makes partial attention outputs exceed device memory. EdenLive and Seko
use the shared V-only adapter; this model keeps its raw-value fallback here.
"""

from lightx2v.common.kvcache.attn.rope import PhysicalRoPERingAttention
from lightx2v.common.kvcache.attn.rope_utils import RotaryFrequencyCache, apply_model_rope
from lightx2v.common.kvcache.attn.sage import SageRingAttention, sage_kv_attention, update_sage_kv_cache
from lightx2v.common.kvcache.attn.sage_rope import SageRoPERingView, _rotate_query
from lightx2v.common.kvcache.rolling import StepRollingKVCachePool


class H3PhysicalRoPERingAttention(PhysicalRoPERingAttention):
    """Reuse a rotated-K scratch tensor across serialized step banks and layers."""

    def _key_workspace_key(self, pool, cache_key, raw):
        if isinstance(pool, StepRollingKVCachePool):
            return (id(pool), tuple(raw.key.shape), raw.key.dtype, raw.key.device)
        return super()._key_workspace_key(pool, cache_key, raw)


class H3SageRoPERingAttention(SageRingAttention):
    """Preserve H3's variable AV writes without one-token V-only segments."""

    def __init__(self, *, rotary_dim=None, rope_layout="interleaved", rope_fma=True, segment_tokens=None, segment_candidates=None, profile="minimax_h3", share_step_keys=True):
        super().__init__(profile=profile, segment_candidates=segment_candidates)
        if rope_layout not in ("interleaved", "split_half"):
            raise ValueError("RoPE layout must be interleaved or split_half")
        if segment_tokens is not None and segment_tokens <= 0:
            raise ValueError("Sage segment_tokens must be positive")
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self.rope_fma = rope_fma
        self.segment_tokens = segment_tokens
        self.share_step_keys = share_step_keys
        self._step_key_workspaces = {}
        self._generations = {}
        self._active_layouts = {}
        self._write_views = {}
        self._frequencies = RotaryFrequencyCache()

    def prepare(self, pool, layer_id, update, *, freqs):
        """Audio/step-cache entry, with a fresh anchor for each output chunk."""
        raw = pool.prepare_attention(layer_id, update)
        cache_key = id(pool), pool._metadata_index(layer_id)
        return self._prepare(pool, layer_id, update, raw, cache_key, freqs, update.global_end)

    def _prepare(self, pool, layer_id, update, raw, cache_key, freqs, generation):
        rotary_dim = self.rotary_dim or raw.key.shape[-1]
        prepared = self._frequencies.prepare(freqs, rotary_dim, self.rope_layout)
        layout = pool._attention_layouts[pool._metadata_index(layer_id)]
        logical_start = update.attention_start if update.attention_start else (0 if layout.active and update.write_end < raw.key.shape[0] else -1)
        segment_tokens = self.select_segment_tokens(
            raw.key,
            write_tokens=update.write_end - update.write_start,
            sink_tokens=update.sink_tokens,
            visible_tokens=update.write_end - update.attention_start,
            default_segment_tokens=self.segment_tokens,
            logical_start=logical_start,
            freqs=prepared,
            rotary_dim=rotary_dim,
            rope_layout=self.rope_layout,
            rope_fma=self.rope_fma,
            refresh_keys=True,
        )
        quant_key = id(pool), pool._metadata_index(layer_id), segment_tokens
        quantized = self.get_quantized_cache(quant_key, raw.key, segment_tokens)
        shared_keys = self.share_step_keys and isinstance(pool, StepRollingKVCachePool)
        if shared_keys:
            workspace_key = id(pool), tuple(raw.key.shape), raw.key.dtype, raw.key.device, segment_tokens
            keys = self._step_key_workspaces.setdefault(workspace_key, (quantized.key_int8, quantized.key_scale, quantized.key_mean))
            if quantized.key_int8 is not keys[0]:
                quantized = quantized._replace(key_int8=keys[0], key_scale=keys[1], key_mean=keys[2])
                signature, _ = self.quantized_layers[quant_key]
                self.quantized_layers[quant_key] = signature, quantized
        layer_key = quant_key[:2]
        if self._active_layouts.get(layer_key) != quant_key:
            # Another chunk layout may have committed newer raw history since
            # this companion was last used. Its historical K and V are stale.
            quantized.valid.zero_()
            quantized.key_valid.zero_()
            self._active_layouts[layer_key] = quant_key
        generation = generation, self._frequencies.last_revision
        if not shared_keys and self._generations.get(quant_key) != generation:
            quantized.key_valid.zero_()
            self._generations[quant_key] = generation
        self._write_views[cache_key] = pool.write_views(layer_id, update.write_start, update.write_end)
        if update.attention_start:
            raw = raw._replace(visible_tokens=update.write_end - update.attention_start)
        return SageRoPERingView(raw, quantized, prepared, logical_start, rotary_dim, self.rope_layout, self.rope_fma, refresh_keys=shared_keys)

    def write_views(self, cache_key):
        return self._write_views[cache_key]

    @staticmethod
    def update_rope_ring(key, value, cache):
        from lightx2v.common.kvcache.kernel import store_kv_ring

        if cache.refresh_keys:
            # Step banks/layers share only rotated K scratch, not raw history.
            # Capture invalidation too, so graph replay never uses another layer.
            cache.quantized.key_valid.zero_()
        store_kv_ring(key, value, cache.raw.key, cache.raw.value, cache.raw.slot_metadata)
        # A clean KV-only commit must refresh the quantized companion too;
        # otherwise the next chunk would reuse the last noisy denoising KV.
        update_sage_kv_cache(
            cache.raw.key,
            cache.raw.value,
            cache.raw.slot_metadata,
            cache.quantized,
            key.shape[0],
            cache.raw.visible_tokens,
            freqs=cache.freqs,
            logical_start=cache.logical_start,
            rotary_dim=cache.rotary_dim or cache.raw.key.shape[-1],
            rope_layout=cache.rope_layout,
            rope_fma=cache.rope_fma,
        )

    def apply_rope_ring(self, query, key, value, cache, *, attention, rope, query_freqs):
        # Current KV must be visible to fused quantization in its physical slots.
        self.update_rope_ring(key, value, cache)
        query = apply_model_rope(rope, query, query_freqs, cache.rotary_dim or query.shape[-1])
        return sage_kv_attention(query, cache.quantized, cache.raw.visible_tokens)

    def apply(self, query, key, value, cache, attention):
        self.update_rope_ring(key, value, cache)
        if cache.rope_layout != "interleaved" or cache.rotary_dim not in (0, query.shape[-1]) or not cache.rope_fma:
            from lightx2v.common.kvcache.kernel import rotate_token_rope

            query = rotate_token_rope(query, cache.freqs, cache.raw.slot_metadata, rotary_dim=cache.rotary_dim, split_half=cache.rope_layout == "split_half", rope_fma=cache.rope_fma)
        else:
            query = _rotate_query(query, cache.freqs, cache.raw.slot_metadata)
        return sage_kv_attention(query, cache.quantized, cache.raw.visible_tokens)

    def reset(self, keep_storage=False):
        super().reset(keep_storage)
        self._generations.clear()
        self._active_layouts.clear()
        self._write_views.clear()
        if not keep_storage:
            self._step_key_workspaces.clear()
            self._frequencies.clear()
