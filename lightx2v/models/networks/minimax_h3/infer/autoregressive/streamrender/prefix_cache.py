"""Request-owned KV bank for StreamRender's fixed conditioning prefix.

Each bank belongs to one guidance branch and denoising timestep. The first
continuation stores the already rotated prefix K/V. Later windows overwrite
only the suffix of the same physical buffer, allowing FlashAttention to read
the prefix and current rows as one view without concatenating either tensor.

Sage keeps immutable prefix rows encoded. Its raw workspace is shared across
serialized layers/banks; only the prefix's partial final segment needs raw K/V
to rebuild that boundary when the current suffix changes.
"""

from typing import NamedTuple

import torch

from lightx2v.common.kvcache.rolling import RollingKVCachePool
from lightx2v.common.kvcache.rolling_offload import OffloadedRollingKVCachePool


class StreamRenderSageCurrent(NamedTuple):
    quantized: object
    metadata: torch.Tensor
    visible_tokens: int

    @property
    def bindings(self):
        return (*self.quantized.bindings, self.metadata)

    def apply(self, query, key, value):
        from lightx2v.common.kvcache.attn.sage import sage_kv_attention, update_sage_kv_cache

        # Fused QKV projections return strided views; the ring quantizer reads
        # dense physical slots. Only this uncached current chunk needs a copy.
        update_sage_kv_cache(key.contiguous(), value.contiguous(), self.metadata, self.quantized, self.visible_tokens, self.visible_tokens)
        return sage_kv_attention(query, self.quantized, self.visible_tokens)


class StreamRenderH3PrefixBank:
    def __init__(self, num_layers, capacity, num_heads, head_dim, dtype, device, *, offload=False, sage=False, segment_tokens=None, segment_candidates=None, raw_workspace=None):
        self.compact_sage = sage and not offload
        if raw_workspace is not None and not self.compact_sage:
            raise ValueError("Shared raw workspaces require resident Sage prefix storage")
        pool_type = OffloadedRollingKVCachePool if offload else RollingKVCachePool
        self.pool = raw_workspace or pool_type(1 if self.compact_sage else num_layers, capacity, num_heads, head_dim, dtype, device)
        self.pool._ensure_kv_buffer()
        if raw_workspace is not None:
            actual = self.pool._key_buffer(0)
            expected_device = torch.device(device)
            if expected_device.type == "cuda" and expected_device.index is None:
                expected_device = torch.device("cuda", torch.cuda.current_device())
            if tuple(actual.shape) != (capacity, num_heads, head_dim) or actual.dtype != dtype or actual.device != expected_device:
                raise ValueError("Shared StreamRender raw workspace geometry must match its bank")
        self.prefix_length = None
        self._captured_layers = set()
        self._prefix_tails = {}
        self.segment_tokens = segment_tokens
        self.sage_attention = None
        if sage:
            from lightx2v.common.kvcache.attn.sage import SageRingAttention

            self.sage_attention = SageRingAttention(profile="streamrender", segment_candidates=segment_candidates)

    def _pool_layer(self, layer_id):
        return 0 if self.compact_sage else layer_id

    def _select_segment(self, write_tokens, visible_tokens):
        if self.segment_tokens is None:
            self.segment_tokens = self.sage_attention.select_segment_tokens(self.pool._key_buffer(0), write_tokens=write_tokens, visible_tokens=visible_tokens, query_tokens=write_tokens)
        return self.segment_tokens

    def prepare_sage(self, layer_id, write_start, write_tokens):
        """Bind quantized physical slots before compile or graph capture.

        Only the immutable prefix is part of the pool's committed length.
        Ephemeral current rows must never enlarge CPU offload prefetches.
        """
        from lightx2v.common.kvcache.attn.sage import SageRingAttentionState
        from lightx2v.common.kvcache.layout import KVCacheUpdate

        end = write_start + write_tokens
        raw = self.pool.prepare_pending_attention(self._pool_layer(layer_id), KVCacheUpdate(write_start, end, 0, end, 0, 0))
        segment_tokens = self._select_segment(write_tokens, end)
        quantized = self.sage_attention.get_quantized_cache((id(self), layer_id, segment_tokens), raw.key, segment_tokens)
        self.pool.upload_write_metadata()
        return SageRingAttentionState(raw.key, raw.value, raw.slot_metadata, end, quantized)

    def prepare_layer(self, layer_id):
        self.pool.begin_layer(self._pool_layer(layer_id))
        tail = self._prefix_tails.get(layer_id)
        if tail is not None and layer_id in self._captured_layers:
            start = self.prefix_length - tail[0].shape[0]
            self.pool._key_buffer(0)[start : self.prefix_length].copy_(tail[0])
            self.pool._value_buffer(0)[start : self.prefix_length].copy_(tail[1])

    def write_views(self, layer_id, suffix_length):
        end = self.prefix_length + suffix_length
        layer_id = self._pool_layer(layer_id)
        return (
            self.pool.k_cache(layer_id, self.prefix_length, end),
            self.pool.v_cache(layer_id, self.prefix_length, end),
        )

    def graph_bindings(self, layer_id):
        layer_id = self._pool_layer(layer_id)
        return self.pool._key_buffer(layer_id), self.pool._value_buffer(layer_id)

    def capture(self, layer_id, key, value):
        """Store post-RoPE prefix rows produced by the first continuation."""
        if self.prefix_length is None:
            self.prefix_length = key.shape[0]
        elif key.shape[0] != self.prefix_length:
            raise ValueError("A StreamRender prefix bank must keep a fixed prefix length")
        pool_layer = self._pool_layer(layer_id)
        self.pool.store_kv(key, value, 0, key.shape[0], pool_layer)
        self.pool.set_ends(pool_layer, key.shape[0], key.shape[0])
        if self.compact_sage:
            segment_tokens = self._select_segment(key.shape[0], key.shape[0])
            tail_tokens = self.prefix_length % segment_tokens
            if tail_tokens:
                previous = self._prefix_tails.get(layer_id)
                if previous is None or previous[0].shape != key[-tail_tokens:].shape:
                    previous = self._prefix_tails[layer_id] = (torch.empty_like(key[-tail_tokens:]), torch.empty_like(value[-tail_tokens:]))
                previous[0].copy_(key[-tail_tokens:])
                previous[1].copy_(value[-tail_tokens:])
            else:
                self._prefix_tails.pop(layer_id, None)
        self._captured_layers.add(layer_id)

    def append_current(self, layer_id, key, value):
        """Write current rows after the immutable prefix and return a KV view."""
        if layer_id not in self._captured_layers:
            raise RuntimeError(f"StreamRender prefix layer {layer_id} has not been captured")
        end = self.prefix_length + key.shape[0]
        if end > self.pool.cache_size:
            raise ValueError(f"StreamRender prefix plus current length {end} exceeds KV capacity {self.pool.cache_size}")
        # The current rows are ephemeral: only the immutable prefix belongs in
        # CPU backing storage. Write the suffix to this layer's GPU staging view.
        if self.compact_sage:
            raise RuntimeError("Encoded StreamRender prefixes require their prepared Sage state")
        self.pool.k_cache(layer_id, self.prefix_length, end).copy_(key)
        self.pool.v_cache(layer_id, self.prefix_length, end).copy_(value)
        return self.pool.k_cache(layer_id, 0, end), self.pool.v_cache(layer_id, 0, end)

    def finish_layer(self, layer_id):
        self.pool.end_layer(self._pool_layer(layer_id))

    def clear(self):
        self.pool.sync_all()
        self.pool.reset()
        self.prefix_length = None
        self._captured_layers.clear()
        if self.sage_attention is not None:
            self.sage_attention.reset(keep_storage=True)


def streamrender_prefix_attention(
    calculate,
    query,
    key,
    value,
    *,
    bank=None,
    layer_id=0,
    prefix_length=0,
    capture_prefix=False,
    prepared_key=None,
    prepared_value=None,
    prepared_sage=None,
):
    """Evaluate source-equivalent prefix visibility over physical KV.

    During capture rows are ordered [prefix, current]. Prefix queries see only
    the prefix; current queries see both. Later calls carry only current rows.
    """
    if prepared_sage is not None:
        if isinstance(prepared_sage, StreamRenderSageCurrent):
            return prepared_sage.apply(query, key, value)
        return bank.sage_attention.apply(query, key, value, prepared_sage, calculate)

    if prepared_key is not None:
        end = prefix_length + key.shape[0]
        prepared_key[prefix_length:end].copy_(key)
        prepared_value[prefix_length:end].copy_(value)
        return calculate.apply(q=query, k=prepared_key[:end], v=prepared_value[:end], causal=False)

    if bank is None:
        return calculate.apply(q=query, k=key, v=value, causal=False)

    if capture_prefix:
        bank.capture(layer_id, key[:prefix_length], value[:prefix_length])
        if bank.sage_attention is not None:
            prefix_state = bank.prepare_sage(layer_id, 0, prefix_length)
            prefix = bank.sage_attention.apply(query[:prefix_length], key[:prefix_length], value[:prefix_length], prefix_state, calculate)
            if prefix_length == query.shape[0]:
                return prefix
            current_state = bank.prepare_sage(layer_id, prefix_length, key.shape[0] - prefix_length)
            current = bank.sage_attention.apply(query[prefix_length:], key[prefix_length:], value[prefix_length:], current_state, calculate)
            return torch.cat((prefix, current), dim=0)
        prefix = calculate.apply(q=query[:prefix_length], k=key[:prefix_length], v=value[:prefix_length], causal=False)
        if prefix_length == query.shape[0]:
            return prefix
        current = calculate.apply(q=query[prefix_length:], k=key, v=value, causal=False)
        result = torch.empty((query.shape[0], prefix.shape[-1]), dtype=prefix.dtype, device=prefix.device)
        result[:prefix_length].copy_(prefix)
        result[prefix_length:].copy_(current)
        return result

    if bank.sage_attention is not None:
        state = bank.prepare_sage(layer_id, bank.prefix_length, key.shape[0])
        return bank.sage_attention.apply(query, key, value, state, calculate)
    key_view, value_view = bank.append_current(layer_id, key, value)
    return calculate.apply(q=query, k=key_view, v=value_view, causal=False)
