from collections.abc import Callable
import torch
from packaging.version import parse

_KV_TORCH_VER = None


def _kvcache_dma_stream_priority() -> int:
    """Match :class:`WeightAsyncStreamManager` ``cuda_load_stream`` so KV DMA
    and weight H2D share the same relative scheduling to the compute stream.
    """
    global _KV_TORCH_VER
    if not torch.cuda.is_available():
        return 0
    if _KV_TORCH_VER is None:
        _KV_TORCH_VER = parse(torch.__version__.split("+")[0])
    return 1 if _KV_TORCH_VER >= parse("2.7") else 0


class KVOffloadPlugin:
    """Double-buffered async CPU↔GPU transfer via CUDA streams (GPU event handoff).

    Pass ``async_load`` / ``async_store`` that perform the actual copies; the
    plugin owns scheduling only.

    Subclasses of cache pools should call :meth:`prefetch_initial` once,
    :meth:`begin_layer` before self-attention, :meth:`end_layer` after, and
    :meth:`sync_all` at the end of the block loop.
    """

    def __init__(
        self,
        device: torch.device,
        async_load: Callable[[int, int], None],
        async_store: Callable[[int, int, int, int], None],
    ) -> None:
        self._device = device
        self._offload_async_load = async_load
        self._offload_async_store = async_store
        # Dedicated streams: H2D / D2H do not use the per-thread default stream, so
        # they can run concurrently with compute kernels (when compute uses another stream).
        pr = _kvcache_dma_stream_priority()
        self._load_stream = torch.cuda.Stream(device=device, priority=pr)
        self._store_stream = torch.cuda.Stream(device=device, priority=pr)
        self._load_done = [torch.cuda.Event() for _ in range(2)]
        self._writeback_done = [torch.cuda.Event() for _ in range(2)]
        cur = torch.cuda.current_stream()
        for e in self._load_done + self._writeback_done:
            e.record(cur)

        self._cur_buf = 0
        self._gpu_layer: list[int] = [-1, -1]
        self._dirty: list[tuple[int, int] | None] = [None, None]

    @property
    def cur_buf(self) -> int:
        return self._cur_buf

    def mark_dirty(self, buf: int, start: int, end: int) -> None:
        if self._dirty[buf] is None:
            self._dirty[buf] = (start, end)
        else:
            s, e = self._dirty[buf]
            self._dirty[buf] = (min(s, start), max(e, end))

    def _issue_prefetch(self, layer_id: int, buf: int) -> None:
        self._load_stream.wait_event(self._writeback_done[buf])
        with torch.cuda.stream(self._load_stream):
            self._offload_async_load(layer_id, buf)
            self._load_done[buf].record(self._load_stream)
        self._gpu_layer[buf] = layer_id
        self._dirty[buf] = None

    def prefetch_initial(self, layer_ids: list[int]) -> None:
        assert len(layer_ids) <= 2
        self._cur_buf = 0
        for buf, lid in enumerate(layer_ids):
            self._issue_prefetch(lid, buf)

    def begin_layer(self, layer_id: int) -> None:
        buf = self._cur_buf
        if self._gpu_layer[buf] != layer_id:
            self._issue_prefetch(layer_id, buf)
        torch.cuda.current_stream().wait_event(self._load_done[buf])

    def end_layer(self, layer_id: int, next_prefetch: int | None = None) -> None:
        buf = self._cur_buf
        dirty = self._dirty[buf]
        if dirty is None:
            self._writeback_done[buf].record(torch.cuda.current_stream())
        else:
            start, end = dirty
            done = torch.cuda.Event()
            done.record()
            self._store_stream.wait_event(done)
            with torch.cuda.stream(self._store_stream):
                self._offload_async_store(layer_id, buf, start, end)
                self._writeback_done[buf].record(self._store_stream)
            self._dirty[buf] = None
        if next_prefetch is not None:
            self._issue_prefetch(next_prefetch, buf)
        self._cur_buf = 1 - self._cur_buf

    def sync_all(self) -> None:
        self._store_stream.synchronize()
        self._load_stream.synchronize()

    def reset_state(self) -> None:
        self._gpu_layer = [-1, -1]
        self._dirty = [None, None]
        self._cur_buf = 0
        cur = torch.cuda.current_stream()
        for e in self._load_done + self._writeback_done:
            e.record(cur)