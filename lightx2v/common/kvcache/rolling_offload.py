"""CPU-backed rolling caches with one shared GPU staging layer."""

import torch

from .layout import KVCacheUpdate
from .offload import KVCacheOffloadMixin, RollingKVOffload
from .rolling import HybridStepRollingKVCachePool, RollingKVCachePool, StepRollingKVCachePool


class OffloadedRollingKVCachePool(KVCacheOffloadMixin, RollingKVCachePool):
    """Keep physical ring storage in pinned memory and stage one layer."""

    def _ensure_kv_buffer(self) -> None:
        if self._offload is None:
            self._init_kv_buffer()

    def _init_kv_buffer(self) -> None:
        self._defer_raw_prefetch = False
        self._quantized_attention_finishes = {}
        self._offload = RollingKVOffload(
            tuple(self.get_max_cache_shape(layer) for layer in range(self._num_layers)),
            self._metadata_shape(),
            self._num_heads,
            self._head_dim,
            self._dtype,
            self._device,
            self._metadata_index,
            self._visible_physical_spans,
            key_only=self.key_only,
        )
        self._init_cache_metadata()

    def prepare_attention(self, layer_id: int, update: KVCacheUpdate):
        self._ensure_kv_buffer()
        self._defer_raw_prefetch = False
        self._quantized_attention_finishes.pop(self._metadata_index(layer_id), None)
        return super().prepare_attention(layer_id, update)

    def prepare_quantized_attention(self, layer_id: int, update: KVCacheUpdate, *, skip_raw_history: bool, on_finish):
        """Delay raw prefetch until the quantized attention policy can decide."""
        self._ensure_kv_buffer()
        self._defer_raw_prefetch = True
        self._quantized_attention_finishes[self._metadata_index(layer_id)] = on_finish
        if skip_raw_history:
            self._offload.begin_layer_without_history(layer_id)
        return super().prepare_attention(layer_id, update)

    def prefetch_initial(self, layer_ids: list[int]) -> None:
        if not getattr(self, "_defer_raw_prefetch", False):
            super().prefetch_initial(layer_ids)

    def end_layer(self, layer_id: int, next_prefetch: int | None = None) -> None:
        # The next layer's eligibility is only known in Sage.prepare. Eagerly
        # staging it here would perform the H2D transfer before that decision.
        if self._defer_raw_prefetch:
            next_prefetch = self._num_layers
        super().end_layer(layer_id, next_prefetch)

    def _visible_physical_spans(self, layer_id: int):
        """Prefetch live ring slots even when they lie above the logical length."""
        return self.ring_layout(layer_id).spans(0, self.get_valid_length(layer_id))

    def _key_buffer(self, layer_id: int) -> torch.Tensor:
        return self._offload.key_buffer[: self.get_max_cache_shape(layer_id)]

    def _value_buffer(self, layer_id: int) -> torch.Tensor:
        return self._offload.value_buffer[: self.get_max_cache_shape(layer_id)]

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, start_idx: int, end_idx: int, layer_id: int) -> None:
        if self.key_only:
            raise RuntimeError("Raw-value writes are unavailable for a quantized-V cache; use its attention backend")
        if end_idx <= start_idx:
            return
        self._check_resident_layer(layer_id)
        layout = self.ring_layout(layer_id)
        self._offload.store(k, v, layer_id, start_idx, layout.spans(start_idx, end_idx))
        layout.record_write(end_idx)

    def finish_attention(self, layer_id: int, update: KVCacheUpdate, *, sink_tokens: int = 0) -> None:
        layout = self._attention_layouts[self._metadata_index(layer_id)]
        self._offload.store_staging(layer_id, layout.spans(update.write_start, update.write_end), sink_tokens)
        on_finish = self._quantized_attention_finishes.pop(self._metadata_index(layer_id), None)
        if on_finish is not None:
            on_finish()
        self.end_layer(layer_id)

    @property
    def allocated_bytes(self) -> int:
        return self._offload.cpu_bytes

    def reset(self) -> None:
        if self._offload is not None:
            self._offload.clear()
            self._defer_raw_prefetch = False
            self._quantized_attention_finishes.clear()
        super().reset()


class OffloadedStepRollingKVCachePool(OffloadedRollingKVCachePool, StepRollingKVCachePool):
    """Stage independent step caches without reusing a stale loaded layer."""

    @StepRollingKVCachePool.current_step.setter
    def current_step(self, value: int) -> None:
        if value == self._current_step:
            return
        if self._offload is not None:
            self.sync_all()
            self._current_step = value
            self._offload.reset()
        else:
            self._current_step = value


class OffloadedHybridStepRollingKVCachePool(OffloadedStepRollingKVCachePool, HybridStepRollingKVCachePool):
    """Stage per-layer capacities while retaining the shared step layout."""
