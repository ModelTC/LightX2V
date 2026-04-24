import torch

from .base import BaseKVCachePool


class RollingKVCachePool(BaseKVCachePool):
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

    def _init_kv_buffer(self):
        super()._init_kv_buffer()
        self._global_end = torch.zeros(self._num_layers, dtype=torch.long, device=self._device)
        self._local_end = torch.zeros(self._num_layers, dtype=torch.long, device=self._device)

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
        layer_id: int,
    ) -> None:
        self._k_buffer[layer_id][start_idx:end_idx] = k
        self._v_buffer[layer_id][start_idx:end_idx] = v

    def get_global_end(self, layer_id: int) -> int:
        return int(self._global_end[layer_id].item())

    def get_local_end(self, layer_id: int) -> int:
        return int(self._local_end[layer_id].item())

    def set_ends(self, layer_id: int, global_end: int, local_end: int) -> None:
        self._global_end[layer_id] = global_end
        self._local_end[layer_id] = local_end

    def roll_window(
        self,
        layer_id: int,
        sink_tokens: int,
        num_evicted: int,
    ) -> None:
        num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
        src_start = sink_tokens + num_evicted
        src_end = src_start + num_kept
        dst_start = sink_tokens
        dst_end = dst_start + num_kept

        kb, vb = self._k_buffer[layer_id], self._v_buffer[layer_id]
        kb[dst_start:dst_end].copy_(kb[src_start:src_end].clone())
        vb[dst_start:dst_end].copy_(vb[src_start:src_end].clone())
