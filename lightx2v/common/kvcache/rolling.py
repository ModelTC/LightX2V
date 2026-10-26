from dataclasses import replace
from itertools import product
from typing import NamedTuple

import torch

from .base import BaseKVCachePool
from .layout import KVCacheUpdate, RingLayout
from .slot import SlotMetadata


class RingKVCacheView(NamedTuple):
    key: torch.Tensor
    value: torch.Tensor
    slot_metadata: torch.Tensor
    visible_tokens: int

    @property
    def bindings(self) -> tuple[torch.Tensor, ...]:
        return self.key, self.value, self.slot_metadata


class RollingKVCachePool(BaseKVCachePool):
    """KV storage with a fixed sink and a ring of recent tokens."""

    def __init__(self, num_layers: int, cache_size: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device)
        self._write_metadata = None
        self._attention_views = {}
        self._attention_layouts = {}
        self.key_only = False
        self.quantized_only = False
        self.value_write_alignment = ()

    def configure_quantized_only(self, *, write_alignment=()) -> None:
        """Keep ring positions only; the pre-RoPE attention owns encoded K/V."""
        if self._k_buffer is not None or bool(getattr(self, "_k_buckets", None)) or getattr(self, "_offload", None) is not None:
            raise RuntimeError("Configure encoded KV storage before allocating the cache")
        self.configure_key_only(write_alignment=write_alignment)
        self.quantized_only = True

    def configure_key_only(self, *, write_alignment=()) -> None:
        """Keep raw K only; the attention backend owns persistent encoded V."""
        alignment = tuple(int(tokens) for tokens in write_alignment if tokens)
        if any(tokens <= 0 for tokens in alignment):
            raise ValueError("Quantized V write alignment must be positive")
        allocated = self._k_buffer is not None or bool(getattr(self, "_k_buckets", None)) or getattr(self, "_offload", None) is not None
        if self.key_only:
            if alignment and self.value_write_alignment != alignment and (self.value_write_alignment or allocated):
                raise ValueError("Cannot change quantized V geometry on an existing pool")
            if alignment:
                self.value_write_alignment = alignment
            return
        if allocated:
            raise RuntimeError("Configure quantized V storage before allocating the cache")
        self.key_only = True
        self.value_write_alignment = alignment

    def _metadata_shape(self) -> tuple[int, ...]:
        return (self._num_layers,)

    def _metadata_index(self, layer_id: int) -> tuple[int, ...]:
        return (layer_id,)

    def _init_cache_metadata(self) -> None:
        self._attention_layouts.clear()
        shape = self._metadata_shape()
        self._global_end = torch.zeros(shape, dtype=torch.long, device="cpu")
        self._local_end = torch.zeros(shape, dtype=torch.long, device="cpu")
        self._ring_layouts = {index: RingLayout(self.get_max_cache_shape(index[-1])) for index in product(*(range(size) for size in shape))}

    def ring_layout(self, layer_id: int) -> RingLayout:
        return self._ring_layouts[self._metadata_index(layer_id)]

    def get_max_cache_shape(self, layer_id: int) -> int:
        return self._cache_size

    def _key_buffer(self, layer_id: int) -> torch.Tensor:
        if self.quantized_only:
            return self._k_buffer
        return self._k_buffer[self._metadata_index(layer_id)]

    def _value_buffer(self, layer_id: int) -> torch.Tensor:
        if self.key_only:
            return self._v_buffer
        return self._v_buffer[self._metadata_index(layer_id)]

    def _init_kv_buffer(self) -> None:
        shape = (*self._metadata_shape(), self._cache_size, self._num_heads, self._head_dim)
        self._k_buffer = torch.empty((0, self._num_heads, self._head_dim) if self.quantized_only else shape, dtype=self._dtype, device=self._device)
        self._v_buffer = torch.empty(0, dtype=self._dtype, device=self._device) if self.key_only else torch.empty_like(self._k_buffer)
        self._init_cache_metadata()

    def _check_resident_layer(self, layer_id: int) -> None:
        pass

    def prefetch_initial(self, layer_ids: list[int]) -> None:
        pass

    def begin_layer(self, layer_id: int) -> None:
        pass

    def end_layer(self, layer_id: int, next_prefetch: int | None = None) -> None:
        pass

    def finish_attention(self, layer_id: int, update: KVCacheUpdate, *, sink_tokens: int = 0) -> None:
        """Persist writes made directly through a physical attention view."""
        pass

    def sync_all(self) -> None:
        pass

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_update: KVCacheUpdate) -> tuple[torch.Tensor, torch.Tensor]:
        """Write a chunk, advance its positions, and return the visible attention KV."""
        self.update_kv(key_states, value_states, layer_idx, cache_update)
        return self.get_kv_cache(layer_idx, cache_update.attention_start, cache_update.write_end)

    def update_kv(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_update: KVCacheUpdate) -> None:
        """Write a chunk and advance its positions without materializing a read."""
        if cache_update.evicted_tokens:
            self.roll_window(layer_idx, cache_update.sink_tokens, cache_update.evicted_tokens)
        self.begin_layer(layer_idx)
        self.store_kv(key_states, value_states, cache_update.write_start, cache_update.write_end, layer_idx)
        self.set_ends(layer_idx, cache_update.global_end, cache_update.write_end)

    def get_kv_cache(self, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache(layer_idx, start, end), self.v_cache(layer_idx, start, end)

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, start_idx: int, end_idx: int, layer_id: int) -> None:
        if self.key_only:
            raise RuntimeError("Raw-value writes are unavailable for a quantized-V cache; use its attention backend")
        if end_idx <= start_idx:
            return
        self._check_resident_layer(layer_id)
        layout = self.ring_layout(layer_id)
        spans = layout.spans(start_idx, end_idx)
        for span in spans:
            source_start = span.logical_start - start_idx
            source_end = source_start + span.length
            physical_end = span.physical_start + span.length
            self._key_buffer(layer_id)[span.physical_start : physical_end].copy_(k[source_start:source_end])
            self._value_buffer(layer_id)[span.physical_start : physical_end].copy_(v[source_start:source_end])
        layout.record_write(end_idx)

    def prepare_attention(self, layer_id: int, update: KVCacheUpdate) -> RingKVCacheView:
        self._ensure_kv_buffer()
        self.initialize_slot_metadata()
        self.begin_layer(layer_id)
        if update.evicted_tokens:
            self.roll_window(layer_id, update.sink_tokens, update.evicted_tokens)
        self.prepare_slot_write(layer_id, update.write_start, update.write_end)
        self.set_ends(layer_id, update.global_end, update.write_end)
        self._attention_layouts[self._metadata_index(layer_id)] = self.ring_layout(layer_id)
        return self._attention_view(layer_id, update.write_end)

    def prepare_pending_attention(self, layer_id: int, update: KVCacheUpdate) -> RingKVCacheView:
        """Prepare a denoising write without committing the cache positions."""
        self._ensure_kv_buffer()
        self.initialize_slot_metadata()
        self.begin_layer(layer_id)
        layout = replace(self.ring_layout(layer_id))
        layout.roll(update.sink_tokens, update.evicted_tokens, self.get_valid_length(layer_id))
        index = self._metadata_index(layer_id)
        self._write_metadata.prepare(index, update.write_start, layout.sink_tokens, layout.recent_head)
        self._attention_layouts[index] = layout
        return self._attention_view(layer_id, update.write_end)

    def _attention_view(self, layer_id: int, visible_tokens: int) -> RingKVCacheView:
        index = self._metadata_index(layer_id)
        key, value = self._key_buffer(layer_id), self._value_buffer(layer_id)
        cached = self._attention_views.get(index)
        if cached is None or cached[0].data_ptr() != key.data_ptr() or cached[1].data_ptr() != value.data_ptr():
            # Independent storage descriptors keep AOT mutation offsets local to the layer.
            cached = torch.from_dlpack(key), torch.from_dlpack(value)
            self._attention_views[index] = cached
        return RingKVCacheView(*cached, self.slot_metadata[index], visible_tokens)

    def write_views(self, layer_id: int, start: int, end: int) -> tuple[torch.Tensor, ...]:
        """Physical write destinations to preserve during graph capture."""
        if self.quantized_only:
            return ()
        self._check_resident_layer(layer_id)
        layout = self._attention_layouts.get(self._metadata_index(layer_id), self.ring_layout(layer_id))
        buffers = (self._key_buffer(layer_id),) if self.key_only else (self._key_buffer(layer_id), self._value_buffer(layer_id))
        return tuple(buffer[span.physical_start : span.physical_start + span.length] for span in layout.spans(start, end) for buffer in buffers)

    def _read_logical(self, buffer: torch.Tensor, layer_id: int, start: int | None, end: int | None, out: torch.Tensor | None) -> torch.Tensor:
        self._check_resident_layer(layer_id)
        if start is None and end is None:
            start, end = 0, self.get_valid_length(layer_id)
        spans = self.ring_layout(layer_id).spans(start, end)
        if out is not None:
            for span in spans:
                offset = span.logical_start - start
                out[offset : offset + span.length].copy_(buffer[span.physical_start : span.physical_start + span.length])
            return out
        if not spans:
            return buffer[:0]
        parts = [buffer[span.physical_start : span.physical_start + span.length] for span in spans]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def k_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
        if self.quantized_only:
            raise RuntimeError("K is encoded in the attention cache; no raw K cache exists")
        return self._read_logical(self._key_buffer(layer_id), layer_id, attn_start, local_end, out)

    def v_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
        if self.key_only:
            raise RuntimeError("V is encoded in the attention cache; no raw V cache exists")
        return self._read_logical(self._value_buffer(layer_id), layer_id, attn_start, local_end, out)

    def initialize_slot_metadata(self) -> None:
        if self._write_metadata is None:
            self._write_metadata = SlotMetadata(self._metadata_shape(), self._device)

    def prepare_slot_write(self, layer_id: int, start: int, end: int) -> None:
        layout = self.ring_layout(layer_id)
        if not 0 <= start <= end <= layout.capacity:
            raise ValueError(f"write [{start}, {end}) exceeds layer {layer_id} capacity")
        self._write_metadata.prepare(self._metadata_index(layer_id), start, layout.sink_tokens, layout.recent_head)
        layout.record_write(end)

    def upload_write_metadata(self) -> None:
        self._write_metadata.upload()

    @property
    def slot_metadata(self) -> torch.Tensor:
        return self._write_metadata.metadata

    def get_seq_length(self, layer_id: int) -> int:
        """Cumulative processed tokens, including tokens evicted from the window."""
        return int(self._global_end[self._metadata_index(layer_id)].item())

    def get_valid_length(self, layer_id: int) -> int:
        """Tokens currently stored in the bounded cache window."""
        return int(self._local_end[self._metadata_index(layer_id)].item())

    def set_ends(self, layer_id: int, global_end: int, local_end: int) -> None:
        index = self._metadata_index(layer_id)
        self._global_end[index] = global_end
        self._local_end[index] = local_end

    def roll_window(self, layer_id: int, sink_tokens: int, num_evicted: int) -> None:
        self.ring_layout(layer_id).roll(sink_tokens, num_evicted, self.get_valid_length(layer_id))

    def reset(self) -> None:
        self._init_cache_metadata()


class StepRollingKVCachePool(RollingKVCachePool):
    """An independent rolling cache for each denoising step."""

    def __init__(self, num_steps: int, num_layers: int, cache_size: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device) -> None:
        self.num_steps = num_steps
        self._current_step = 0
        super().__init__(num_layers, cache_size, num_heads, head_dim, dtype, device)

    @property
    def current_step(self) -> int:
        return self._current_step

    @current_step.setter
    def current_step(self, value: int) -> None:
        if value == self._current_step:
            return
        self._current_step = value

    def _metadata_shape(self) -> tuple[int, ...]:
        return self.num_steps, self._num_layers

    def _metadata_index(self, layer_id: int) -> tuple[int, ...]:
        return self._current_step, layer_id


class HybridStepRollingKVCachePool(StepRollingKVCachePool):
    """Step caches grouped by per-layer capacity to avoid padding allocations."""

    def __init__(self, num_steps: int, num_layers: int, layer_cache_sizes: list[int] | tuple[int, ...], num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        sizes = tuple(layer_cache_sizes)
        if len(sizes) != num_layers:
            raise ValueError(f"layer_cache_sizes must contain {num_layers} entries, got {len(sizes)}")
        if not sizes or min(sizes) <= 0:
            raise ValueError("layer_cache_sizes must contain positive token capacities")
        self._k_buckets = {}
        self._v_buckets = {}
        self._layer_cache_sizes = sizes
        self._layer_bucket_slots = []
        self._bucket_counts = {}
        for size in sizes:
            slot = self._bucket_counts.get(size, 0)
            self._layer_bucket_slots.append((size, slot))
            self._bucket_counts[size] = slot + 1
        super().__init__(num_steps, num_layers, max(sizes), num_heads, head_dim, dtype, device)

    @property
    def layer_cache_sizes(self) -> tuple[int, ...]:
        return self._layer_cache_sizes

    @property
    def bucket_counts(self) -> dict[int, int]:
        return dict(self._bucket_counts)

    def get_max_cache_shape(self, layer_id: int) -> int:
        return self._layer_cache_sizes[layer_id]

    def _key_buffer(self, layer_id: int) -> torch.Tensor:
        if self.quantized_only:
            return self._k_buffer
        capacity, slot = self._layer_bucket_slots[layer_id]
        return self._k_buckets[capacity][self._current_step, slot]

    def _value_buffer(self, layer_id: int) -> torch.Tensor:
        if self.key_only:
            return self._v_buffer
        capacity, slot = self._layer_bucket_slots[layer_id]
        return self._v_buckets[capacity][self._current_step, slot]

    def _ensure_kv_buffer(self) -> None:
        allocated = self._k_buffer is not None if self.quantized_only else bool(self._k_buckets)
        if not allocated:
            self._init_kv_buffer()

    def _allocate_buckets(self) -> dict[int, torch.Tensor]:
        return {capacity: torch.empty(self.num_steps, count, capacity, self._num_heads, self._head_dim, dtype=self._dtype, device=self._device) for capacity, count in self._bucket_counts.items()}

    def _init_kv_buffer(self) -> None:
        if self.quantized_only:
            self._k_buffer = torch.empty((0, self._num_heads, self._head_dim), dtype=self._dtype, device=self._device)
            self._v_buffer = torch.empty_like(self._k_buffer)
            self._init_cache_metadata()
            return
        self._k_buckets = self._allocate_buckets()
        self._v_buckets = {} if self.key_only else self._allocate_buckets()
        if self.key_only:
            self._v_buffer = torch.empty(0, dtype=self._dtype, device=self._device)
        self._init_cache_metadata()

    @property
    def allocated_bytes(self) -> int:
        return sum(buffer.nbytes for buffer in (*self._k_buckets.values(), *self._v_buckets.values()))
