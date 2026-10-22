"""Request-owned KV bank for StreamRender's fixed conditioning prefix.

Each bank belongs to one guidance branch and denoising timestep. The first
continuation stores the already rotated prefix K/V. Later windows overwrite
only the suffix of the same physical buffer, allowing dense attention to read
the prefix and current rows as one view without concatenating either tensor.
"""

import torch

from lightx2v.common.kvcache.rolling import RollingKVCachePool
from lightx2v.common.kvcache.rolling_offload import OffloadedRollingKVCachePool


class StreamRenderH3PrefixBank:
    def __init__(self, num_layers, capacity, num_heads, head_dim, dtype, device, *, offload=False):
        pool_type = OffloadedRollingKVCachePool if offload else RollingKVCachePool
        self.pool = pool_type(num_layers, capacity, num_heads, head_dim, dtype, device)
        self.pool._ensure_kv_buffer()
        self.prefix_length = None
        self._captured_layers = set()

    def prepare_layer(self, layer_id):
        self.pool.begin_layer(layer_id)

    def write_views(self, layer_id, suffix_length):
        end = self.prefix_length + suffix_length
        return (
            self.pool.k_cache(layer_id, self.prefix_length, end),
            self.pool.v_cache(layer_id, self.prefix_length, end),
        )

    def graph_bindings(self, layer_id):
        return self.pool._key_buffer(layer_id), self.pool._value_buffer(layer_id)

    def capture(self, layer_id, key, value):
        """Store post-RoPE prefix rows produced by the first continuation."""
        if self.prefix_length is None:
            self.prefix_length = key.shape[0]
        elif key.shape[0] != self.prefix_length:
            raise ValueError("A StreamRender prefix bank must keep a fixed prefix length")
        self.pool.store_kv(key, value, 0, key.shape[0], layer_id)
        self.pool.set_ends(layer_id, key.shape[0], key.shape[0])
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
        self.pool.k_cache(layer_id, self.prefix_length, end).copy_(key)
        self.pool.v_cache(layer_id, self.prefix_length, end).copy_(value)
        return self.pool.k_cache(layer_id, 0, end), self.pool.v_cache(layer_id, 0, end)

    def finish_layer(self, layer_id):
        self.pool.end_layer(layer_id)

    def clear(self):
        self.pool.sync_all()
        self.pool.reset()
        self.prefix_length = None
        self._captured_layers.clear()


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
):
    """Evaluate source-equivalent prefix visibility with dense attention.

    During capture rows are ordered [prefix, current]. Prefix queries see only
    the prefix; current queries see both. Later calls carry only current rows.
    """
    if prepared_key is not None:
        end = prefix_length + key.shape[0]
        prepared_key[prefix_length:end].copy_(key)
        prepared_value[prefix_length:end].copy_(value)
        return calculate.apply(q=query, k=prepared_key[:end], v=prepared_value[:end], causal=False)

    if bank is None:
        return calculate.apply(q=query, k=key, v=value, causal=False)

    if capture_prefix:
        bank.capture(layer_id, key[:prefix_length], value[:prefix_length])
        prefix = calculate.apply(q=query[:prefix_length], k=key[:prefix_length], v=value[:prefix_length], causal=False)
        if prefix_length == query.shape[0]:
            return prefix
        current = calculate.apply(q=query[prefix_length:], k=key, v=value, causal=False)
        result = torch.empty((query.shape[0], prefix.shape[-1]), dtype=prefix.dtype, device=prefix.device)
        result[:prefix_length].copy_(prefix)
        result[prefix_length:].copy_(current)
        return result

    key_view, value_view = bank.append_current(layer_id, key, value)
    return calculate.apply(q=query, k=key_view, v=value_view, causal=False)
