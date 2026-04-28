import torch
from loguru import logger

from .base import BaseKVCachePool
from .offload import KVOffloadPlugin


class RollingKVCachePool(BaseKVCachePool):
    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        kv_offload: bool = False,
    ) -> None:
        self._kv_offload = kv_offload
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device)

    def _init_kv_buffer(self):
        if self._kv_offload:
            self._init_kv_buffer_offload()
            return
        super()._init_kv_buffer()
        self._global_end = torch.zeros(self._num_layers, dtype=torch.long, device=self._device)
        self._local_end = torch.zeros(self._num_layers, dtype=torch.long, device=self._device)

    def _init_kv_buffer_offload(self) -> None:
        L, N, H, D = self._num_layers, self._cache_size, self._num_heads, self._head_dim
        self._k_cpu = torch.zeros(L, N, H, D, dtype=self._dtype, device="cpu").pin_memory()
        self._v_cpu = torch.zeros(L, N, H, D, dtype=self._dtype, device="cpu").pin_memory()
        self._k_gpu_buf = torch.zeros(2, N, H, D, dtype=self._dtype, device=self._device)
        self._v_gpu_buf = torch.zeros(2, N, H, D, dtype=self._dtype, device=self._device)
        self._global_end = torch.zeros(L, dtype=torch.long, device=self._device)
        self._local_end = torch.zeros(L, dtype=torch.long, device=self._device)

        def _async_load(layer_id: int, buf: int) -> None:
            self._k_gpu_buf[buf].copy_(self._k_cpu[layer_id], non_blocking=True)
            self._v_gpu_buf[buf].copy_(self._v_cpu[layer_id], non_blocking=True)

        def _async_store(layer_id: int, buf: int, start: int, end: int) -> None:
            self._k_cpu[layer_id, start:end].copy_(
                self._k_gpu_buf[buf, start:end],
                non_blocking=True,
            )
            self._v_cpu[layer_id, start:end].copy_(
                self._v_gpu_buf[buf, start:end],
                non_blocking=True,
            )

        self._offload = KVOffloadPlugin(self._device, _async_load, _async_store)
        gpu_mb = (self._k_gpu_buf.nbytes + self._v_gpu_buf.nbytes) / (1024 * 1024)
        cpu_mb = (self._k_cpu.nbytes + self._v_cpu.nbytes) / (1024 * 1024)
        logger.info(
            "[RollingKVCachePool+offload] GPU fixed buffer: {:.1f} MB, CPU pinned: {:.1f} MB (saved {:.1f} MB GPU)",
            gpu_mb,
            cpu_mb,
            cpu_mb - gpu_mb,
        )

    def _mark_offload_dirty(self, start_idx: int, end_idx: int) -> None:
        self._offload.mark_dirty(self._offload.cur_buf, start_idx, end_idx)

    def prefetch_initial(self, layer_ids: list[int]) -> None:
        if not self._kv_offload:
            return
        self._offload.prefetch_initial(layer_ids)

    def begin_layer(self, layer_id: int) -> None:
        if not self._kv_offload:
            return
        self._offload.begin_layer(layer_id)

    def end_layer(self, layer_id: int, next_prefetch: int | None = None) -> None:
        if not self._kv_offload:
            return
        self._offload.end_layer(layer_id, next_prefetch=next_prefetch)

    def sync_all(self) -> None:
        if not self._kv_offload:
            return
        self._offload.sync_all()

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
        layer_id: int,
    ) -> None:
        if not self._kv_offload:
            self._k_buffer[layer_id][start_idx:end_idx] = k
            self._v_buffer[layer_id][start_idx:end_idx] = v
            return
        buf = self._offload.cur_buf
        self._k_gpu_buf[buf, start_idx:end_idx] = k
        self._v_gpu_buf[buf, start_idx:end_idx] = v
        self._mark_offload_dirty(start_idx, end_idx)

    def k_cache(
        self,
        layer_id: int,
        attn_start: int | None = None,
        local_end: int | None = None,
    ) -> torch.Tensor:
        if not self._kv_offload:
            if attn_start is None and local_end is None:
                return self._k_buffer[layer_id]
            return self._k_buffer[layer_id][attn_start:local_end]
        if attn_start is None and local_end is None:
            return self._k_gpu_buf[self._offload.cur_buf]
        return self._k_gpu_buf[self._offload.cur_buf, attn_start:local_end]

    def v_cache(
        self,
        layer_id: int,
        attn_start: int | None = None,
        local_end: int | None = None,
    ) -> torch.Tensor:
        if not self._kv_offload:
            if attn_start is None and local_end is None:
                return self._v_buffer[layer_id]
            return self._v_buffer[layer_id][attn_start:local_end]
        if attn_start is None and local_end is None:
            return self._v_gpu_buf[self._offload.cur_buf]
        return self._v_gpu_buf[self._offload.cur_buf, attn_start:local_end]

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
        if not self._kv_offload:
            num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
            src_start = sink_tokens + num_evicted
            src_end = src_start + num_kept
            dst_start = sink_tokens
            dst_end = dst_start + num_kept
            kb, vb = self._k_buffer[layer_id], self._v_buffer[layer_id]
            kb[dst_start:dst_end].copy_(kb[src_start:src_end].clone())
            vb[dst_start:dst_end].copy_(vb[src_start:src_end].clone())
            return

        num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
        src_s = sink_tokens + num_evicted
        dst_s = sink_tokens
        buf = self._offload.cur_buf
        kb = self._k_gpu_buf[buf]
        vb = self._v_gpu_buf[buf]
        kb[dst_s : dst_s + num_kept].copy_(kb[src_s : src_s + num_kept].clone())
        vb[dst_s : dst_s + num_kept].copy_(vb[src_s : src_s + num_kept].clone())
        self._mark_offload_dirty(dst_s, dst_s + num_kept)

    def reset(self) -> None:
        if not self._kv_offload:
            self._k_buffer.zero_()
            self._v_buffer.zero_()
            return
        self._k_cpu.zero_()
        self._v_cpu.zero_()
        self._k_gpu_buf.zero_()
        self._v_gpu_buf.zero_()
        self._global_end.zero_()
        self._local_end.zero_()
        self._offload.reset_state()