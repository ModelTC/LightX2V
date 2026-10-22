"""Transfer ordering for one GPU staging layer backed by CPU KV storage."""

from collections.abc import Callable
from itertools import product

import torch
from loguru import logger
from packaging.version import parse

from .layout import RingSpan


class KVCacheOffload:
    """Share staging storage while keeping writes and layer loads ordered."""

    def __init__(self, num_layers, event_shape, device, copy_layer, layer_index):
        self.num_layers = num_layers
        self.device = device
        self.copy_layer = copy_layer
        self.layer_index = layer_index
        priority = 1 if parse(torch.__version__.split("+")[0]) >= parse("2.7") else 0
        self.stream = torch.cuda.Stream(device=device, priority=priority)
        self.load_done = torch.cuda.Event()
        self.staging_free = torch.cuda.Event()
        self.write_done = {index: torch.cuda.Event() for index in product(*(range(size) for size in event_shape))}
        self.reset()

    def reset(self):
        self.loaded_layer = -1
        current_stream = torch.cuda.current_stream(self.device)
        for event in (self.load_done, self.staging_free, *self.write_done.values()):
            event.record(current_stream)

    def record_write(self, layer):
        self.write_done[self.layer_index(layer)].record(torch.cuda.current_stream(self.device))

    def prefetch(self, layer):
        if layer >= self.num_layers:
            return
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.write_done[self.layer_index(layer)])
            self.stream.wait_event(self.staging_free)
            self.copy_layer(layer)
            self.load_done.record(self.stream)
        self.loaded_layer = layer

    def begin_layer(self, layer):
        if self.loaded_layer != layer:
            self.prefetch(layer)
        torch.cuda.current_stream(self.device).wait_event(self.load_done)

    def end_layer(self, layer, next_prefetch=None):
        self.staging_free.record(torch.cuda.current_stream(self.device))
        self.prefetch(layer + 1 if next_prefetch is None else next_prefetch)

    def check_layer(self, layer):
        if self.loaded_layer != layer:
            raise RuntimeError(f"cache layer {layer} requested but GPU staging holds layer {self.loaded_layer}")

    def synchronize(self):
        self.stream.synchronize()
        torch.cuda.current_stream(self.device).synchronize()


class KVCacheOffloadMixin:
    """Expose staging lifecycle for caches with optional CPU backing."""

    def __init__(self, *args, **kwargs) -> None:
        self._offload = None
        super().__init__(*args, **kwargs)

    def _check_resident_layer(self, layer_id: int) -> None:
        if self._offload is not None:
            self._offload.check_layer(layer_id)

    def prefetch_initial(self, layer_ids: list[int]) -> None:
        if self._offload is not None:
            self._offload.prefetch(layer_ids[0] if layer_ids else 0)

    def begin_layer(self, layer_id: int) -> None:
        if self._offload is not None:
            self._offload.begin_layer(layer_id)

    def end_layer(self, layer_id: int, next_prefetch: int | None = None) -> None:
        if self._offload is not None:
            self._offload.end_layer(layer_id, next_prefetch)

    def sync_all(self) -> None:
        if self._offload is not None:
            self._offload.synchronize()


class RollingKVOffload(KVCacheOffload):
    """Own CPU rolling-cache storage and the GPU layer used for attention."""

    def __init__(
        self,
        layer_capacities: tuple[int, ...],
        metadata_shape: tuple[int, ...],
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_index: Callable[[int], tuple[int, ...]],
        visible_spans: Callable[[int], list[RingSpan]],
    ) -> None:
        self.visible_spans = visible_spans
        self._layer_slots = []
        bucket_counts = {}
        for capacity in layer_capacities:
            slot = bucket_counts.get(capacity, 0)
            self._layer_slots.append((capacity, slot))
            bucket_counts[capacity] = slot + 1
        self._key_cpu = {capacity: torch.empty((*metadata_shape[:-1], count, capacity, num_heads, head_dim), dtype=dtype, device="cpu", pin_memory=True) for capacity, count in bucket_counts.items()}
        self._value_cpu = {capacity: torch.empty_like(buffer, pin_memory=True) for capacity, buffer in self._key_cpu.items()}
        staging_shape = (max(layer_capacities), num_heads, head_dim)
        self.key_buffer = torch.empty(staging_shape, dtype=dtype, device=device)
        self.value_buffer = torch.empty_like(self.key_buffer)
        super().__init__(len(layer_capacities), metadata_shape, device, self._load_layer, layer_index)
        logger.info("[RollingKVOffload] GPU staging: {:.1f} MiB, CPU pinned KV: {:.1f} MiB", self.gpu_bytes / 2**20, self.cpu_bytes / 2**20)

    @property
    def cpu_bytes(self) -> int:
        return sum(buffer.nbytes for buffer in (*self._key_cpu.values(), *self._value_cpu.values()))

    @property
    def gpu_bytes(self) -> int:
        return self.key_buffer.nbytes + self.value_buffer.nbytes

    def _cpu_buffers(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        capacity, slot = self._layer_slots[layer]
        index = (*self.layer_index(layer)[:-1], slot)
        return self._key_cpu[capacity][index], self._value_cpu[capacity][index]

    def _load_layer(self, layer: int) -> None:
        key_cpu, value_cpu = self._cpu_buffers(layer)
        for span in self.visible_spans(layer):
            physical = slice(span.physical_start, span.physical_start + span.length)
            self.key_buffer[physical].copy_(key_cpu[physical], non_blocking=True)
            self.value_buffer[physical].copy_(value_cpu[physical], non_blocking=True)

    def store(self, key: torch.Tensor, value: torch.Tensor, layer: int, logical_start: int, spans: list[RingSpan]) -> None:
        key_cpu, value_cpu = self._cpu_buffers(layer)
        for span in spans:
            source_start = span.logical_start - logical_start
            source_end = source_start + span.length
            physical_end = span.physical_start + span.length
            self.key_buffer[span.physical_start : physical_end].copy_(key[source_start:source_end])
            self.value_buffer[span.physical_start : physical_end].copy_(value[source_start:source_end])
            key_cpu[span.physical_start : physical_end].copy_(key[source_start:source_end], non_blocking=True)
            value_cpu[span.physical_start : physical_end].copy_(value[source_start:source_end], non_blocking=True)
        self.record_write(layer)

    def store_staging(self, layer: int, spans: list[RingSpan], sink_tokens: int = 0) -> None:
        """Persist physical kernel writes, including a refreshed sink RoPE anchor."""
        key_cpu, value_cpu = self._cpu_buffers(layer)
        ranges = [(span.physical_start, span.physical_start + span.length) for span in spans]
        if sink_tokens:
            ranges.append((0, sink_tokens))
        for start, end in ranges:
            key_cpu[start:end].copy_(self.key_buffer[start:end], non_blocking=True)
            value_cpu[start:end].copy_(self.value_buffer[start:end], non_blocking=True)
        self.record_write(layer)

    def clear(self) -> None:
        self.synchronize()
        for buffer in (*self._key_cpu.values(), *self._value_cpu.values(), self.key_buffer, self.value_buffer):
            buffer.zero_()
        self.reset()
