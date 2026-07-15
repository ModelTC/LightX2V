import torch

from .base import BaseKVCachePool
from .utils import _kvcache_dma_stream_priority


class RollingKVCachePool(BaseKVCachePool):
    """Rolling KV cache implemented as sink-fixed + recent-ring buffer.

    Logical layout exposed to callers is unchanged:
        [0 : local_end) == [sink fixed][recent logical window]

    Physical layout after the first roll becomes:
        [sink fixed][recent ring]

    ``roll_window`` never copies the kept recent window. It only advances the
    recent ring head and shrinks recent_len. ``k_cache`` / ``v_cache`` return a
    contiguous logical tensor, concatenating at most two recent ring fragments.

    """

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

    # ---------------------------------------------------------------------
    # Ring metadata helpers
    # ---------------------------------------------------------------------
    def _ring_shape(self):
        return (self._num_layers,)

    def _meta_idx(self, layer_id: int):
        return int(layer_id)

    def _init_ring_metadata(self) -> None:
        shape = self._ring_shape()
        self._ring_active = torch.zeros(shape, dtype=torch.bool, device="cpu")
        self._ring_sink = torch.zeros(shape, dtype=torch.long, device="cpu")
        self._ring_head = torch.zeros(shape, dtype=torch.long, device="cpu")
        self._ring_recent_len = torch.zeros(shape, dtype=torch.long, device="cpu")

    def _is_ring_active(self, layer_id: int) -> bool:
        return bool(self._ring_active[self._meta_idx(layer_id)].item())

    def _ring_get(self, layer_id: int) -> tuple[bool, int, int, int]:
        idx = self._meta_idx(layer_id)
        return (
            bool(self._ring_active[idx].item()),
            int(self._ring_sink[idx].item()),
            int(self._ring_head[idx].item()),
            int(self._ring_recent_len[idx].item()),
        )

    def _ring_set(self, layer_id: int, *, active: bool, sink: int, head: int, recent_len: int) -> None:
        idx = self._meta_idx(layer_id)
        self._ring_active[idx] = bool(active)
        self._ring_sink[idx] = int(sink)
        self._ring_head[idx] = int(head)
        self._ring_recent_len[idx] = int(recent_len)

    def _ensure_ring_active(self, layer_id: int, sink_tokens: int) -> None:
        active, sink, head, recent_len = self._ring_get(layer_id)
        local_end = self.get_local_end(layer_id)
        if active:
            if sink != int(sink_tokens):
                raise RuntimeError(f"ring sink changed for layer {layer_id}: old={sink}, new={sink_tokens}")
            return
        sink = int(sink_tokens)
        if sink < 0 or sink >= self._cache_size:
            raise RuntimeError(f"invalid sink_tokens={sink}, cache_size={self._cache_size}")
        recent_len = max(0, int(local_end) - sink)
        if recent_len > self._cache_size - sink:
            raise RuntimeError("recent_len exceeds recent ring capacity")
        self._ring_set(layer_id, active=True, sink=sink, head=0, recent_len=recent_len)

    def _recent_capacity(self, sink: int) -> int:
        cap = self._cache_size - int(sink)
        if cap <= 0:
            raise RuntimeError(f"invalid recent capacity: cache_size={self._cache_size}, sink={sink}")
        return cap

    def _logical_chunks(self, layer_id: int, start: int, end: int) -> list[tuple[int, int, int]]:
        """Return logical->physical chunks as ``(logical_start, physical_start, length)``.

        Chunks are returned in logical order. They address the physical full CPU
        cache layout, not the GPU offload window.
        """
        start, end = int(start), int(end)
        if end <= start:
            return []
        active, sink, head, recent_len = self._ring_get(layer_id)
        if not active:
            return [(start, start, end - start)]

        chunks: list[tuple[int, int, int]] = []
        # Fixed sink part.
        if start < sink:
            s0, e0 = start, min(end, sink)
            if e0 > s0:
                chunks.append((s0, s0, e0 - s0))

        # Recent ring part. Logical recent offset is compacted after rolling:
        # logical token sink + off maps to physical sink + ((head + off) % cap).
        r0 = max(start, sink)
        r1 = end
        if r1 > r0:
            cap = self._recent_capacity(sink)
            off = r0 - sink
            length = r1 - r0
            if off + length > cap:
                raise RuntimeError(f"logical range [{start}, {end}) exceeds ring capacity; sink={sink}, cap={cap}")
            pos = (head + off) % cap
            first = min(length, cap - pos)
            chunks.append((r0, sink + pos, first))
            if first < length:
                chunks.append((r0 + first, sink, length - first))
        return chunks

    # ---------------------------------------------------------------------
    # Buffer accessors. Step/spatial variants override these.
    # ---------------------------------------------------------------------
    def _k_layer(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_gpu_buf
        return self._k_buffer[layer_id]

    def _v_layer(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_gpu_buf
        return self._v_buffer[layer_id]

    def _k_cpu_layer(self, layer_id: int) -> torch.Tensor:
        return self._k_cpu[layer_id]

    def _v_cpu_layer(self, layer_id: int) -> torch.Tensor:
        return self._v_cpu[layer_id]

    # ---------------------------------------------------------------------
    # Init / offload setup
    # ---------------------------------------------------------------------
    def _init_kv_buffer(self):
        if self._kv_offload:
            self._init_kv_buffer_offload()
            return
        super()._init_kv_buffer()
        self._global_end = torch.zeros(self._num_layers, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(self._num_layers, dtype=torch.long, device="cpu")
        self._init_ring_metadata()

    def _init_kv_buffer_offload(self) -> None:
        from loguru import logger

        L, N, H, D = self._num_layers, self._cache_size, self._num_heads, self._head_dim
        d = self._device

        # CPU pinned buffers hold the authoritative per-layer physical ring; a
        # single GPU staging buffer holds the layer currently being computed.
        self._k_cpu = torch.empty(L, N, H, D, dtype=self._dtype, device="cpu").pin_memory()
        self._v_cpu = torch.empty(L, N, H, D, dtype=self._dtype, device="cpu").pin_memory()
        self._k_gpu_buf = torch.empty(N, H, D, dtype=self._dtype, device=d)
        self._v_gpu_buf = torch.empty(N, H, D, dtype=self._dtype, device=d)

        self._global_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._init_ring_metadata()
        self._init_offload_state((L,))

        gpu_mb = (self._k_gpu_buf.nbytes + self._v_gpu_buf.nbytes) / (1024 * 1024)
        cpu_mb = (self._k_cpu.nbytes + self._v_cpu.nbytes) / (1024 * 1024)
        logger.info(
            "[{}+offload] GPU staging layer: {:.1f} MB, CPU pinned: {:.1f} MB",
            self.__class__.__name__,
            gpu_mb,
            cpu_mb,
        )

    def _make_event_tree(self, shape: tuple[int, ...]):
        if len(shape) == 1:
            return [torch.cuda.Event() for _ in range(shape[0])]
        return [self._make_event_tree(shape[1:]) for _ in range(shape[0])]

    def _flatten_events(self, events) -> list[torch.cuda.Event]:
        out: list[torch.cuda.Event] = []
        for event in events:
            if isinstance(event, torch.cuda.Event):
                out.append(event)
            else:
                out.extend(self._flatten_events(event))
        return out

    def _offload_index(self, layer_id: int) -> tuple[int, ...]:
        return (int(layer_id),)

    def _cpu_update_event(self, layer_id: int) -> torch.cuda.Event:
        event = self._cpu_update_done
        for idx in self._offload_index(layer_id):
            event = event[idx]
        return event

    def _offload_events(self) -> list[torch.cuda.Event]:
        return [self._load_done, self._staging_free, *self._flatten_events(self._cpu_update_done)]

    def _init_offload_state(self, event_shape: tuple[int, ...]) -> None:
        pr = _kvcache_dma_stream_priority()
        self._prefetch_stream = torch.cuda.Stream(device=self._device, priority=pr)
        self._load_done = torch.cuda.Event()
        # Single GPU staging buffer is shared across layers; this event marks
        # that the compute stream has finished reading the currently-loaded
        # layer, so the prefetch stream may safely overwrite the buffer.
        self._staging_free = torch.cuda.Event()
        self._cpu_update_done = self._make_event_tree(event_shape)
        self._loaded_layer = -1
        cur = torch.cuda.current_stream()
        for event in self._offload_events():
            event.record(cur)

    def _reset_offload_state(self) -> None:
        self._loaded_layer = -1
        cur = torch.cuda.current_stream()
        for event in self._offload_events():
            event.record(cur)

    def _record_cpu_update(self, layer_id: int) -> None:
        self._cpu_update_event(layer_id).record(torch.cuda.current_stream())

    def _copy_layer_to_gpu(self, layer_id: int) -> None:
        self._k_gpu_buf.copy_(self._k_cpu_layer(layer_id), non_blocking=True)
        self._v_gpu_buf.copy_(self._v_cpu_layer(layer_id), non_blocking=True)

    def _prefetch_layer(self, layer_id: int) -> None:
        if layer_id >= self._num_layers:
            return
        with torch.cuda.stream(self._prefetch_stream):
            # CPU authoritative buffer for this layer must be up to date, and
            # the compute stream must be done reading the staging buffer's
            # previous contents before we overwrite the single shared buffer.
            self._prefetch_stream.wait_event(self._cpu_update_event(layer_id))
            self._prefetch_stream.wait_event(self._staging_free)
            self._copy_layer_to_gpu(layer_id)
            self._load_done.record(self._prefetch_stream)
        self._loaded_layer = int(layer_id)

    def _check_layer_loaded(self, layer_id: int) -> None:
        if self._kv_offload and self._loaded_layer != int(layer_id):
            raise RuntimeError(f"cache layer {layer_id} requested but GPU buffer holds layer {self._loaded_layer}")

    def prefetch_initial(self, layer_ids: list[int]) -> None:
        if not self._kv_offload:
            return
        first_layer = int(layer_ids[0]) if layer_ids else 0
        self._prefetch_layer(first_layer)

    def begin_layer(self, layer_id: int) -> None:
        """Wait until the single GPU staging buffer holds ``layer_id``."""
        if not self._kv_offload:
            return
        layer_id = int(layer_id)
        if self._loaded_layer != layer_id:
            self._prefetch_layer(layer_id)
        torch.cuda.current_stream().wait_event(self._load_done)

    def end_layer(self, layer_id: int, next_prefetch: int | None = None) -> None:
        """CPU cache is updated directly by ``store_kv``; prefetch the next layer."""
        if not self._kv_offload:
            return
        # All compute-stream work that reads the staging buffer for this layer
        # (store + attention) is now enqueued; let the prefetch overwrite it.
        self._staging_free.record(torch.cuda.current_stream())
        next_layer = int(layer_id) + 1 if next_prefetch is None else int(next_prefetch)
        self._prefetch_layer(next_layer)

    def sync_all(self) -> None:
        if not self._kv_offload:
            return
        self._prefetch_stream.synchronize()
        torch.cuda.current_stream().synchronize()

    # ---------------------------------------------------------------------
    # Store / read / roll
    # ---------------------------------------------------------------------
    def _update_ring_len_after_store(self, layer_id: int, end_idx: int) -> None:
        active, sink, head, recent_len = self._ring_get(layer_id)
        if not active:
            return
        if end_idx > sink:
            new_recent_len = max(recent_len, int(end_idx) - sink)
            cap = self._recent_capacity(sink)
            if new_recent_len > cap:
                raise RuntimeError(f"recent ring overflow: recent_len={new_recent_len}, capacity={cap}")
            self._ring_set(layer_id, active=True, sink=sink, head=head, recent_len=new_recent_len)

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int,
        end_idx: int,
        layer_id: int,
    ) -> None:
        if end_idx <= start_idx:
            return
        self._check_layer_loaded(layer_id)
        kb, vb = self._k_layer(layer_id), self._v_layer(layer_id)
        kcpu = self._k_cpu_layer(layer_id) if self._kv_offload else None
        vcpu = self._v_cpu_layer(layer_id) if self._kv_offload else None
        for logical_s, phys_s, n in self._logical_chunks(layer_id, start_idx, end_idx):
            ks = logical_s - start_idx
            ke = ks + n
            kb[phys_s : phys_s + n].copy_(k[ks:ke])
            vb[phys_s : phys_s + n].copy_(v[ks:ke])
            if self._kv_offload:
                kcpu[phys_s : phys_s + n].copy_(k[ks:ke], non_blocking=True)
                vcpu[phys_s : phys_s + n].copy_(v[ks:ke], non_blocking=True)
        self._update_ring_len_after_store(layer_id, end_idx)
        if self._kv_offload:
            self._record_cpu_update(layer_id)

    def _read_logical(self, layer_id: int, attn_start: int, local_end: int, which: str) -> torch.Tensor:
        self._check_layer_loaded(layer_id)
        base = self._k_layer(layer_id) if which == "k" else self._v_layer(layer_id)
        chunks = self._logical_chunks(layer_id, attn_start, local_end)
        if not chunks:
            return torch.empty(0, self._num_heads, self._head_dim, device=self._device, dtype=self._dtype)
        parts = [base[p : p + n] for _, p, n in chunks]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def k_cache(
        self,
        layer_id: int,
        attn_start: int | None = None,
        local_end: int | None = None,
    ) -> torch.Tensor:
        if attn_start is None and local_end is None:
            attn_start, local_end = 0, self.get_local_end(layer_id)
        return self._read_logical(layer_id, int(attn_start), int(local_end), "k")

    def v_cache(
        self,
        layer_id: int,
        attn_start: int | None = None,
        local_end: int | None = None,
    ) -> torch.Tensor:
        if attn_start is None and local_end is None:
            attn_start, local_end = 0, self.get_local_end(layer_id)
        return self._read_logical(layer_id, int(attn_start), int(local_end), "v")

    def get_global_end(self, layer_id: int) -> int:
        return int(self._global_end[layer_id].item())

    def get_local_end(self, layer_id: int) -> int:
        return int(self._local_end[layer_id].item())

    def set_ends(self, layer_id: int, global_end: int, local_end: int) -> None:
        self._global_end[layer_id] = int(global_end)
        self._local_end[layer_id] = int(local_end)

    def roll_window(self, layer_id: int, sink_tokens: int, num_evicted: int) -> None:
        if num_evicted <= 0:
            return
        self._ensure_ring_active(layer_id, sink_tokens)
        active, sink, head, recent_len = self._ring_get(layer_id)
        if num_evicted > recent_len:
            raise RuntimeError(f"cannot evict {num_evicted} recent tokens, only {recent_len} available")
        cap = self._recent_capacity(sink)
        head = (head + int(num_evicted)) % cap
        recent_len -= int(num_evicted)
        self._ring_set(layer_id, active=True, sink=sink, head=head, recent_len=recent_len)

    def reset(self) -> None:
        if self._kv_offload:
            self.sync_all()
            self._k_cpu.zero_()
            self._v_cpu.zero_()
            self._k_gpu_buf.zero_()
            self._v_gpu_buf.zero_()
            self._global_end.zero_()
            self._local_end.zero_()
            self._init_ring_metadata()
            self._reset_offload_state()
            return
        self._global_end.zero_()
        self._local_end.zero_()
        self._init_ring_metadata()


class StepRollingKVCachePool(RollingKVCachePool):
    """Step-isolated FP rolling KV cache using per-step ring metadata."""

    def __init__(
        self,
        num_steps: int,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        kv_offload: bool = False,
    ) -> None:
        self.num_steps = int(num_steps)
        self._current_step = 0
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device, kv_offload=kv_offload)

    @property
    def current_step(self) -> int:
        return self._current_step

    @current_step.setter
    def current_step(self, value: int) -> None:
        value = int(value)
        if value == self._current_step:
            return
        if self._kv_offload and hasattr(self, "_prefetch_stream"):
            self.sync_all()
            self._current_step = value
            self._reset_offload_state()
            return
        self._current_step = value

    def _step(self) -> int:
        return int(self._current_step)

    def _ring_shape(self):
        return (self.num_steps, self._num_layers)

    def _meta_idx(self, layer_id: int):
        return (self._step(), int(layer_id))

    def _k_layer(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._k_gpu_buf
        return self._k_buffer[self._step(), layer_id]

    def _v_layer(self, layer_id: int) -> torch.Tensor:
        if self._kv_offload:
            return self._v_gpu_buf
        return self._v_buffer[self._step(), layer_id]

    def _k_cpu_layer(self, layer_id: int) -> torch.Tensor:
        return self._k_cpu[self._step(), layer_id]

    def _v_cpu_layer(self, layer_id: int) -> torch.Tensor:
        return self._v_cpu[self._step(), layer_id]

    def _offload_index(self, layer_id: int) -> tuple[int, ...]:
        return (self._step(), int(layer_id))

    def _init_kv_buffer(self) -> None:
        if self._kv_offload:
            self._init_kv_buffer_offload()
            return
        S, L, N, H, D = self.num_steps, self._num_layers, self._cache_size, self._num_heads, self._head_dim
        self._k_buffer = torch.empty(S, L, N, H, D, dtype=self._dtype, device=self._device)
        self._v_buffer = torch.empty(S, L, N, H, D, dtype=self._dtype, device=self._device)
        self._global_end = torch.zeros(S, L, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(S, L, dtype=torch.long, device="cpu")
        self._init_ring_metadata()

    def _init_kv_buffer_offload(self) -> None:
        from loguru import logger

        S, L, N, H, D = self.num_steps, self._num_layers, self._cache_size, self._num_heads, self._head_dim
        d = self._device

        self._k_cpu = torch.empty(S, L, N, H, D, dtype=self._dtype, device="cpu").pin_memory()
        self._v_cpu = torch.empty(S, L, N, H, D, dtype=self._dtype, device="cpu").pin_memory()
        self._k_gpu_buf = torch.empty(N, H, D, dtype=self._dtype, device=d)
        self._v_gpu_buf = torch.empty(N, H, D, dtype=self._dtype, device=d)

        self._global_end = torch.zeros(S, L, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(S, L, dtype=torch.long, device="cpu")
        self._init_ring_metadata()
        self._init_offload_state((S, L))

        gpu_mb = (self._k_gpu_buf.nbytes + self._v_gpu_buf.nbytes) / (1024 * 1024)
        cpu_mb = (self._k_cpu.nbytes + self._v_cpu.nbytes) / (1024 * 1024)
        logger.info(
            "[{}+offload] steps={}, GPU staging layer: {:.1f} MB, CPU pinned: {:.1f} MB",
            self.__class__.__name__,
            self.num_steps,
            gpu_mb,
            cpu_mb,
        )

    def get_global_end(self, layer_id: int) -> int:
        return int(self._global_end[self._step(), layer_id].item())

    def get_local_end(self, layer_id: int) -> int:
        return int(self._local_end[self._step(), layer_id].item())

    def set_ends(self, layer_id: int, global_end: int, local_end: int) -> None:
        self._global_end[self._step(), layer_id] = int(global_end)
        self._local_end[self._step(), layer_id] = int(local_end)


class SpatialRollingKVCachePool(RollingKVCachePool):
    """Spatial FP rolling KV cache with ring along token axis N.

    Buffer shape is [L, S, N, H, D]. The ring metadata is still per layer and
    applies to the N axis for every spatial row.
    """

    def __init__(
        self,
        spatial_len: int,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        kv_offload: bool = False,
    ) -> None:
        if kv_offload:
            raise ValueError("SpatialRollingKVCachePool does not support kv_offload.")
        self._spatial_len = int(spatial_len)
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device, kv_offload=kv_offload)

    @property
    def spatial_len(self) -> int:
        return self._spatial_len

    def _init_kv_buffer(self) -> None:
        L, S, N, H, D = self._num_layers, self._spatial_len, self._cache_size, self._num_heads, self._head_dim
        self._k_buffer = torch.empty(L, S, N, H, D, dtype=self._dtype, device=self._device)
        self._v_buffer = torch.empty(L, S, N, H, D, dtype=self._dtype, device=self._device)
        self._global_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(L, dtype=torch.long, device="cpu")
        self._init_ring_metadata()

    def _k_layer(self, layer_id: int) -> torch.Tensor:
        return self._k_buffer[layer_id]

    def _v_layer(self, layer_id: int) -> torch.Tensor:
        return self._v_buffer[layer_id]

    def _k_cpu_layer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("SpatialRollingKVCachePool does not support kv_offload.")

    def _v_cpu_layer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("SpatialRollingKVCachePool does not support kv_offload.")

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, start_idx: int, end_idx: int, layer_id: int) -> None:
        if end_idx <= start_idx:
            return
        kb, vb = self._k_layer(layer_id), self._v_layer(layer_id)
        for logical_s, phys_s, n in self._logical_chunks(layer_id, start_idx, end_idx):
            ks = logical_s - start_idx
            ke = ks + n
            kb[:, phys_s : phys_s + n].copy_(k[:, ks:ke])
            vb[:, phys_s : phys_s + n].copy_(v[:, ks:ke])
        self._update_ring_len_after_store(layer_id, end_idx)

    def _read_logical(self, layer_id: int, attn_start: int, local_end: int, which: str) -> torch.Tensor:
        base = self._k_layer(layer_id) if which == "k" else self._v_layer(layer_id)
        chunks = self._logical_chunks(layer_id, attn_start, local_end)
        if not chunks:
            return torch.empty(self._spatial_len, 0, self._num_heads, self._head_dim, device=self._device, dtype=self._dtype)
        parts = [base[:, p : p + n] for _, p, n in chunks]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    def k_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None) -> torch.Tensor:
        if attn_start is None and local_end is None:
            attn_start, local_end = 0, self.get_local_end(layer_id)
        return self._read_logical(layer_id, int(attn_start), int(local_end), "k")

    def v_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None) -> torch.Tensor:
        if attn_start is None and local_end is None:
            attn_start, local_end = 0, self.get_local_end(layer_id)
        return self._read_logical(layer_id, int(attn_start), int(local_end), "v")
