import torch

from .base import BaseKVCachePool


class StaticKVCachePool(BaseKVCachePool):
    """Fixed per-layer K/V populated once and reused without eviction.

    This cache is intended for immutable reference streams.  It deliberately
    has no rolling position metadata: ``reset`` invalidates every layer and a
    subsequent ``store_kv`` replaces that layer from offset zero.
    """

    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device)
        self._init_kv_buffer()
        self._valid_lens = torch.zeros(num_layers, dtype=torch.long, device="cpu")

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, layer_id: int) -> None:
        if k.shape != v.shape:
            raise ValueError(f"Static K/V shapes must match, got k={tuple(k.shape)}, v={tuple(v.shape)}")
        if k.ndim != 3:
            raise ValueError(f"Static K/V must have shape [seq, heads, dim], got {tuple(k.shape)}")
        if k.shape[0] > self._cache_size:
            raise ValueError(f"Static K/V length {k.shape[0]} exceeds cache_size={self._cache_size}")
        if tuple(k.shape[1:]) != (self._num_heads, self._head_dim):
            raise ValueError(f"Static K/V head shape mismatch: expected {(self._num_heads, self._head_dim)}, got {tuple(k.shape[1:])}")
        super().store_kv(k, v, layer_id)
        self._valid_lens[layer_id] = k.shape[0]

    def _slice(self, buffer: torch.Tensor, layer_id: int, attn_start, local_end):
        valid_len = int(self._valid_lens[layer_id])
        start = 0 if attn_start is None else int(attn_start)
        end = valid_len if local_end is None else min(int(local_end), valid_len)
        return buffer[layer_id, start:end]

    def k_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None) -> torch.Tensor:
        return self._slice(self._k_buffer, layer_id, attn_start, local_end)

    def v_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None) -> torch.Tensor:
        return self._slice(self._v_buffer, layer_id, attn_start, local_end)

    def is_ready(self, layer_id: int | None = None) -> bool:
        if layer_id is None:
            return bool(torch.all(self._valid_lens > 0))
        return bool(self._valid_lens[int(layer_id)] > 0)

    def reset(self) -> None:
        self._valid_lens.zero_()
