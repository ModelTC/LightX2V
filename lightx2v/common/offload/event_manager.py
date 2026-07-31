import torch

from lightx2v.common.offload.block_slab import carve_block_slab, copy_block_slab_
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class EventSlotWeightAsyncStreamManager(WeightAsyncStreamManager):
    """Weight offload with reusable buffers protected by device events."""

    _EVENT_SLOT_COUNT = 2

    def __init__(self, offload_granularity, load_stream=None, compute_stream=None):
        if offload_granularity != "block":
            raise ValueError("Event-slot weight offload only supports block granularity")

        super().__init__(offload_granularity)
        if load_stream is not None:
            self.cuda_load_stream = load_stream
        if compute_stream is not None:
            self.compute_stream = compute_stream

        self.device_module = torch_device_module
        self._ready_events = [torch_device_module.Event() for _ in range(self._EVENT_SLOT_COUNT)]
        self._free_events = [torch_device_module.Event() for _ in range(self._EVENT_SLOT_COUNT)]
        self.reset_slots()

    @property
    def slot_count(self):
        return self._EVENT_SLOT_COUNT

    def reset_slots(self):
        """Reset slot bookkeeping; synchronize pending device work first."""
        self._slot_pending = [False] * self.slot_count
        self._slot_ready_waited = [False] * self.slot_count
        self._slot_free_recorded = [False] * self.slot_count

    def _validate_slot(self, slot_idx):
        if not isinstance(slot_idx, int) or not 0 <= slot_idx < self.slot_count:
            raise IndexError(f"slot_idx must be in [0, {self.slot_count}), got {slot_idx!r}")
        if not hasattr(self, "cuda_buffers"):
            raise RuntimeError("init_cuda_buffer must be called before using an event slot")
        if len(self.cuda_buffers) < self.slot_count:
            raise RuntimeError(f"Event-slot weight offload requires {self.slot_count} device buffers")

    def _load_block_to_buffer(self, target_buffer, block_idx, blocks, adapter_block_idx):
        block_slab = getattr(self, "block_slabs", {}).get(block_idx)
        if block_slab is not None:
            copy_block_slab_(
                self.block_slab_staging_raw,
                block_slab.raw,
                nbytes=block_slab.layout.nbytes,
                non_blocking=True,
            )
            target_buffer.load_state_dict(
                self.block_slab_device_views[block_idx],
                block_idx,
                adapter_block_idx,
            )
            return

        if hasattr(self, "cpu_buffers"):
            source = self.cpu_buffers[0]
        else:
            if blocks is None:
                raise ValueError("blocks must be provided when CPU buffers have not been initialized")
            source = blocks[block_idx]
        target_buffer.load_state_dict(source.state_dict(), block_idx, adapter_block_idx)

    def prefetch_to_slot(self, slot_idx, block_idx, blocks=None, adapter_block_idx=None):
        """Enqueue one block copy into a fixed staging slot."""
        self._validate_slot(slot_idx)
        if self._slot_pending[slot_idx]:
            raise RuntimeError(f"Offload slot {slot_idx} is still pending; call record_free before reusing it")

        with torch_device_module.stream(self.cuda_load_stream):
            if self._slot_free_recorded[slot_idx]:
                self.cuda_load_stream.wait_event(self._free_events[slot_idx])
            self._load_block_to_buffer(
                self.cuda_buffers[slot_idx],
                block_idx,
                blocks,
                adapter_block_idx,
            )
            self._ready_events[slot_idx].record(self.cuda_load_stream)

        self._slot_pending[slot_idx] = True
        self._slot_ready_waited[slot_idx] = False
        return self.cuda_buffers[slot_idx]

    def wait_ready(self, slot_idx, stream=None):
        """Make the compute stream wait until a slot is ready."""
        self._validate_slot(slot_idx)
        if not self._slot_pending[slot_idx]:
            raise RuntimeError(f"Offload slot {slot_idx} has not been prefetched")

        stream = self.compute_stream if stream is None else stream
        stream.wait_event(self._ready_events[slot_idx])
        self._slot_ready_waited[slot_idx] = True
        return self.cuda_buffers[slot_idx]

    def record_free(self, slot_idx, stream=None):
        """Record that compute has finished consuming a slot."""
        self._validate_slot(slot_idx)
        if not self._slot_pending[slot_idx]:
            raise RuntimeError(f"Offload slot {slot_idx} has not been prefetched")
        if not self._slot_ready_waited[slot_idx]:
            raise RuntimeError(f"wait_ready must be called before record_free for offload slot {slot_idx}")

        stream = self.compute_stream if stream is None else stream
        self._free_events[slot_idx].record(stream)
        self._slot_pending[slot_idx] = False
        self._slot_ready_waited[slot_idx] = False
        self._slot_free_recorded[slot_idx] = True

    def init_block_slabs(self, block_slabs, staging_raw=None):
        """Prepare the shared device buffer used for block-slab copies."""
        block_slabs = dict(block_slabs or {})
        self.block_slabs = block_slabs
        self.block_slab_device_views = {}
        if not block_slabs:
            self.block_slab_staging_raw = None
            return None

        max_nbytes = max(slab.layout.nbytes for slab in block_slabs.values())
        if staging_raw is None:
            staging_raw = torch.empty((max_nbytes,), dtype=torch.uint8, device=AI_DEVICE)
        elif not isinstance(staging_raw, torch.Tensor) or staging_raw.dtype != torch.uint8 or staging_raw.dim() != 1 or not staging_raw.is_contiguous() or staging_raw.numel() < max_nbytes:
            raise ValueError(f"shared block slab staging buffer must be a contiguous uint8 tensor with at least {max_nbytes} bytes")

        self.block_slab_staging_raw = staging_raw
        self.block_slab_device_views = {block_idx: carve_block_slab(staging_raw, slab.layout) for block_idx, slab in block_slabs.items()}
        return staging_raw
