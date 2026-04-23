"""KV cache pools with CPU offloading.

Only 2 layers' worth of KV data resides on GPU at any time (double-buffered).
Async CPU↔GPU transfers via dedicated CUDA streams overlap with compute:

  Per-layer timeline (steady state):
  ┌──────────────────────────────────────────────────────────────────────┐
  │ load_stream:   [pf i+1.....]   [pf i+2.................]            │
  │ compute:       [self-attn i] [cross+ffn i] [self-attn i+1] [cross+] │
  │ store_stream:                [wb i.....]               [wb i+1.....]│
  └──────────────────────────────────────────────────────────────────────┘

  - prefetch (i+2) is queued at end_layer(i) so it starts during cross+ffn i
    (PCIe Gen4 has dual DMA engines — H2D & D2H run concurrently)
  - writeback (i) is also queued at end_layer(i)

Usage:
  - ``prefetch_initial([0, 1])`` once before the loop (preloads first 2 layers)
  - ``begin_layer(layer_id)`` before self-attention (just waits via GPU event)
  - ``end_layer(layer_id, next_prefetch=i+2)`` after self-attention
    (writeback + queue prefetch — both overlap with cross-attn + FFN)
  - ``sync_all()`` after the last layer
"""

import torch
from loguru import logger

from .kernel import quant_value_per_channel_fp8_static_scale_kernel
from .quant import _FP8_MAX, QuantRollingKVCachePool
from .rolling import RollingKVCachePool

# ====================================================================== #
#  Mixin: double-buffered CPU↔GPU transfer logic
# ====================================================================== #


class _KVCacheOffloadMixin:
    """Double-buffered async CPU↔GPU KV cache transfer via CUDA streams.

    Pure GPU-side event handoff — no CPU-blocking sync inside the loop:

      load_stream:   [load 0] [load 1] ─────── [load 2] ─────── ...
                              ↓               ↓
      compute:                [self-attn 0]   [self-attn 1]      ...
                              ↓               ↓
      store_stream:           [wb 0] ─────── [wb 1] ─────────── ...

    Subclasses must implement:
        _offload_async_load(layer_id, buf)              – CPU→GPU full copy
        _offload_async_store(layer_id, buf, start, end) – GPU→CPU partial copy
    """

    def _init_offload(self):
        self._load_stream = torch.cuda.Stream()
        self._store_stream = torch.cuda.Stream()
        # Per-buffer events for fine-grained dependency tracking
        self._load_done = [torch.cuda.Event() for _ in range(2)]
        self._writeback_done = [torch.cuda.Event() for _ in range(2)]
        # Seed events as "completed" so first wait_event is a no-op
        cur = torch.cuda.current_stream()
        for e in self._load_done + self._writeback_done:
            e.record(cur)

        self._cur_buf = 0
        self._gpu_layer = [-1, -1]
        # Per-buffer dirty token range (start, end), inclusive-exclusive.
        # None = clean (no need to writeback). Updated by store_kv / roll_window.
        self._dirty: list[tuple[int, int] | None] = [None, None]

    def _mark_dirty(self, buf: int, start: int, end: int) -> None:
        if self._dirty[buf] is None:
            self._dirty[buf] = (start, end)
        else:
            s, e = self._dirty[buf]
            self._dirty[buf] = (min(s, start), max(e, end))

    # ------------------------------------------------------------------ #

    def _issue_prefetch(self, layer_id: int, buf: int) -> None:
        """Queue an async H2D load of *layer_id* into *buf* on load_stream.

        Waits for any in-flight writeback of the same buf before overwriting.
        """
        self._load_stream.wait_event(self._writeback_done[buf])
        with torch.cuda.stream(self._load_stream):
            self._offload_async_load(layer_id, buf)
            self._load_done[buf].record(self._load_stream)
        self._gpu_layer[buf] = layer_id
        self._dirty[buf] = None  # fresh from CPU → in sync

    def prefetch_initial(self, layer_ids: list[int]) -> None:
        """Pre-fill GPU buffers before the loop starts.

        Pass at most 2 layer ids — typically ``[0, 1]``. Subsequent prefetches
        are issued automatically by ``end_layer(next_prefetch=...)``.
        """
        assert len(layer_ids) <= 2
        self._cur_buf = 0
        for buf, lid in enumerate(layer_ids):
            self._issue_prefetch(lid, buf)

    def begin_layer(self, layer_id: int):
        """Wait (GPU-side) until *layer_id*'s KV is loaded into the active buffer.

        Falls back to issuing a load if *layer_id* wasn't prefetched (e.g. on
        the very first call without ``prefetch_initial``). No CPU block.
        """
        buf = self._cur_buf

        if self._gpu_layer[buf] != layer_id:
            self._issue_prefetch(layer_id, buf)

        torch.cuda.current_stream().wait_event(self._load_done[buf])

    def end_layer(self, layer_id: int, next_prefetch: int | None = None):
        """Queue (a) writeback of the active buffer and (b) the *next* prefetch
        — both run on dedicated streams in parallel with subsequent compute.

        Writeback only transfers the dirty token range (modified by store_kv /
        roll_window). The next prefetch is queued *here* (not in the next
        iteration's begin_layer) so it can start the moment writeback finishes
        — letting it overlap with cross-attn + FFN of the current layer.
        """
        buf = self._cur_buf
        dirty = self._dirty[buf]

        if dirty is None:
            # Nothing to write back — mark writeback as done immediately so a
            # subsequent prefetch into this buf doesn't wait unnecessarily
            self._writeback_done[buf].record(torch.cuda.current_stream())
        else:
            start, end = dirty
            done = torch.cuda.Event()
            done.record()  # captures compute stream's progress
            self._store_stream.wait_event(done)
            with torch.cuda.stream(self._store_stream):
                self._offload_async_store(layer_id, buf, start, end)
                self._writeback_done[buf].record(self._store_stream)
            self._dirty[buf] = None

        # Queue the next prefetch into the buf we're about to free.
        # PCIe Gen4 has independent H2D / D2H DMA engines, so the upcoming
        # prefetch (H2D) can start in parallel with the writeback (D2H) of the
        # OTHER buffer — and crucially before the next iteration's CPU-side
        # kernel launches block load_stream from being scheduled.
        if next_prefetch is not None:
            self._issue_prefetch(next_prefetch, buf)

        self._cur_buf = 1 - self._cur_buf

    def sync_all(self):
        """Block until all outstanding transfers complete (call after the loop)."""
        self._store_stream.synchronize()
        self._load_stream.synchronize()


# ====================================================================== #
#  bf16 rolling KV cache with CPU offload
# ====================================================================== #


class OffloadRollingKVCachePool(_KVCacheOffloadMixin, RollingKVCachePool):
    """RollingKVCachePool with CPU offload — only 2 layers on GPU."""

    def __init__(self, num_layers, cache_size, num_heads, head_dim, dtype, device):
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device)

    def _init_kv_buffer(self):
        L, N, H, D = self._num_layers, self._cache_size, self._num_heads, self._head_dim

        # CPU pinned memory — ground-truth storage for all layers
        self._k_cpu = torch.zeros(L, N, H, D, dtype=self._dtype, device="cpu").pin_memory()
        self._v_cpu = torch.zeros(L, N, H, D, dtype=self._dtype, device="cpu").pin_memory()

        # GPU — fixed contiguous double buffers (2 layers only)
        self._k_gpu_buf = torch.zeros(2, N, H, D, dtype=self._dtype, device=self._device)
        self._v_gpu_buf = torch.zeros(2, N, H, D, dtype=self._dtype, device=self._device)

        self._global_end = torch.zeros(L, dtype=torch.long, device=self._device)
        self._local_end = torch.zeros(L, dtype=torch.long, device=self._device)

        self._init_offload()

        gpu_mb = (self._k_gpu_buf.nbytes + self._v_gpu_buf.nbytes) / (1024 * 1024)
        cpu_mb = (self._k_cpu.nbytes + self._v_cpu.nbytes) / (1024 * 1024)
        logger.info(
            "[OffloadRollingKVCachePool] GPU fixed buffer: {:.1f} MB, CPU pinned: {:.1f} MB (saved {:.1f} MB GPU)",
            gpu_mb,
            cpu_mb,
            cpu_mb - gpu_mb,
        )

    # ------------------------------------------------------------------ #
    #  offload copy helpers
    # ------------------------------------------------------------------ #

    def _offload_async_load(self, layer_id, buf):
        self._k_gpu_buf[buf].copy_(self._k_cpu[layer_id], non_blocking=True)
        self._v_gpu_buf[buf].copy_(self._v_cpu[layer_id], non_blocking=True)

    def _offload_async_store(self, layer_id, buf, start, end):
        self._k_cpu[layer_id, start:end].copy_(
            self._k_gpu_buf[buf, start:end],
            non_blocking=True,
        )
        self._v_cpu[layer_id, start:end].copy_(
            self._v_gpu_buf[buf, start:end],
            non_blocking=True,
        )

    # ------------------------------------------------------------------ #
    #  KV access (redirected to GPU double buffers)
    # ------------------------------------------------------------------ #

    def store_kv(self, k, v, start_idx, end_idx, layer_id):
        buf = self._cur_buf
        self._k_gpu_buf[buf, start_idx:end_idx] = k
        self._v_gpu_buf[buf, start_idx:end_idx] = v
        self._mark_dirty(buf, start_idx, end_idx)

    def k_cache(self, layer_id, attn_start, local_end):
        return self._k_gpu_buf[self._cur_buf, attn_start:local_end]

    def v_cache(self, layer_id, attn_start, local_end):
        return self._v_gpu_buf[self._cur_buf, attn_start:local_end]

    def roll_window(self, layer_id, sink_tokens, num_evicted):
        buf = self._cur_buf
        num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
        src_s = sink_tokens + num_evicted
        dst_s = sink_tokens

        kb = self._k_gpu_buf[buf]
        vb = self._v_gpu_buf[buf]
        kb[dst_s : dst_s + num_kept].copy_(kb[src_s : src_s + num_kept].clone())
        vb[dst_s : dst_s + num_kept].copy_(vb[src_s : src_s + num_kept].clone())

        # roll shifts data within the GPU buffer — CPU is now stale at [dst, dst+num_kept]
        self._mark_dirty(buf, dst_s, dst_s + num_kept)

    def reset(self):
        self._k_cpu.zero_()
        self._v_cpu.zero_()
        self._k_gpu_buf.zero_()
        self._v_gpu_buf.zero_()
        self._global_end.zero_()
        self._local_end.zero_()
        self._gpu_layer = [-1, -1]
        self._dirty = [None, None]
        self._cur_buf = 0


# ====================================================================== #
#  Quantized (K int8 + V fp8) rolling KV cache with CPU offload
# ====================================================================== #
class OffloadQuantRollingKVCachePool(_KVCacheOffloadMixin, QuantRollingKVCachePool):
    """QuantRollingKVCachePool with CPU offload — only 2 layers on GPU.

    K (int8) and V (fp8) bulk data live on CPU in pinned memory; GPU
    keeps a 2-buffer rolling window. Calibration data (km, v_scale,
    k_block_scale) is small and stays on GPU permanently — both store_kv
    and ``k_cache`` / ``v_cache`` (with window args) look up calibrated
    scales by ``(step, layer)`` and
    apply them on the active GPU buffer.

    Inherits ``_quant_key``, K/V pack layout, and ``_roll_window_on_k_v`` from
    :class:`QuantRollingKVCachePool` while routing bulk data through
    :meth:`_load_calib` and the double ``_[kv]_gpu_buf`` workspace.

    Because ``k_block_scale`` is loaded once into ``_calib_k_block_scale``,
    we no longer need to ship per-layer K-scale buffers between CPU and
    GPU (saves both memory and DMA bandwidth).
    """

    # ------------------------------------------------------------------ #
    #  buffer init
    # ------------------------------------------------------------------ #

    def _init_kv_buffer(self) -> None:
        L, N, H, D = self._num_layers, self._cache_size, self._num_heads, self._head_dim
        self._load_calib()

        # CPU pinned memory — ground truth for K (int8) and V (fp8 viewed as uint8)
        self._k_cpu = torch.zeros(L, N, H, D, dtype=torch.int8, device="cpu").pin_memory()
        self._v_cpu = torch.zeros(L, N, H, D, dtype=torch.uint8, device="cpu").pin_memory()

        # GPU — fixed contiguous double buffers (2 layers only)
        self._k_gpu_buf = torch.zeros(2, N, H, D, dtype=torch.int8, device=self._device)
        self._v_gpu_buf = torch.zeros(2, N, H, D, dtype=torch.float8_e4m3fn, device=self._device)

        self._global_end = torch.zeros(L, dtype=torch.long, device=self._device)
        self._local_end = torch.zeros(L, dtype=torch.long, device=self._device)

        self._init_offload()

        gpu_mb = (self._k_gpu_buf.nbytes + self._v_gpu_buf.nbytes) / (1024 * 1024)
        cpu_mb = (self._k_cpu.nbytes + self._v_cpu.nbytes) / (1024 * 1024)
        logger.info(
            "[OffloadQuantRollingKVCachePool] GPU fixed buffer: {:.1f} MB, CPU pinned: {:.1f} MB (saved {:.1f} MB GPU)",
            gpu_mb,
            cpu_mb,
            cpu_mb - gpu_mb,
        )

    def _offload_async_load(self, layer_id, buf):
        self._k_gpu_buf[buf].copy_(self._k_cpu[layer_id], non_blocking=True)
        self._v_gpu_buf[buf].view(torch.uint8).copy_(self._v_cpu[layer_id], non_blocking=True)

    def _offload_async_store(self, layer_id, buf, start, end):
        self._k_cpu[layer_id, start:end].copy_(
            self._k_gpu_buf[buf, start:end],
            non_blocking=True,
        )
        v_gpu_slice_u8 = self._v_gpu_buf[buf, start:end].view(torch.uint8)
        self._v_cpu[layer_id, start:end].copy_(v_gpu_slice_u8, non_blocking=True)

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
        layer_id: int,
    ) -> None:
        buf = self._cur_buf
        km = self._lookup_km(layer_id)
        if km is not None:
            km_lowp = km.to(k.dtype).squeeze(0)
            k_smoothed = k - km_lowp
        else:
            k_smoothed = k

        blk_start = start_idx // self._BLKK
        last_blk = (end_idx - 1) // self._BLKK
        num_blk = last_blk - blk_start + 1
        preset_scale = self._lookup_k_block_scale(layer_id, blk_start, num_blk)
        k_int8 = self._quant_key(k_smoothed, preset_scale, start_idx, self._BLKK)
        self._k_gpu_buf[buf, start_idx:end_idx] = k_int8

        v_scale = self._lookup_v_scale(layer_id)
        v_fp8 = quant_value_per_channel_fp8_static_scale_kernel(v, v_scale, fp8_max=_FP8_MAX)
        self._v_gpu_buf[buf, start_idx:end_idx] = v_fp8

        self._mark_dirty(buf, start_idx, end_idx)

    # ------------------------------------------------------------------ #
    #  read (sage_attn2_kvquant) — same tuple layout as QuantRollingKVCachePool
    # ------------------------------------------------------------------ #

    def k_cache(self, layer_id: int, attn_start: int, local_end: int):
        BLK = self._BLKK
        buf = self._cur_buf
        aligned_start = (attn_start // BLK) * BLK
        k_int8 = self._k_gpu_buf[buf, aligned_start:local_end].unsqueeze(0).contiguous()
        blk_s = aligned_start // BLK
        blk_e = (local_end + BLK - 1) // BLK
        k_scale = self._calib_k_block_scale[self.current_step, layer_id, blk_s:blk_e].permute(1, 0, 2).reshape(1, self._num_heads, -1).contiguous()
        return k_int8, k_scale

    def v_cache(self, layer_id: int, attn_start: int, local_end: int):
        BLK = self._BLKK
        buf = self._cur_buf
        aligned_start = (attn_start // BLK) * BLK
        v_fp8 = self._v_gpu_buf[buf, aligned_start:local_end]
        v_fp8 = self._transpose_permute_v(v_fp8)
        v_scale = self._lookup_v_scale(layer_id).unsqueeze(0).contiguous()
        return v_fp8, v_scale

    # ------------------------------------------------------------------ #
    #  roll
    # ------------------------------------------------------------------ #

    def roll_window(self, layer_id: int, sink_tokens: int, num_evicted: int) -> None:
        buf = self._cur_buf
        self._roll_window_on_k_v(
            self._k_gpu_buf[buf],
            self._v_gpu_buf[buf],
            layer_id,
            sink_tokens,
            num_evicted,
        )
        num_kept = int(self._local_end[layer_id].item()) - num_evicted - sink_tokens
        dst_s = sink_tokens
        self._mark_dirty(buf, dst_s, dst_s + num_kept)

    # ------------------------------------------------------------------ #
    #  misc
    # ------------------------------------------------------------------ #

    def reset(self):
        self._k_cpu.zero_()
        self._v_cpu.zero_()
        self._k_gpu_buf.zero_()
        self._v_gpu_buf.zero_()
        self._global_end.zero_()
        self._local_end.zero_()
        self._gpu_layer = [-1, -1]
        self._dirty = [None, None]
        self._cur_buf = 0
