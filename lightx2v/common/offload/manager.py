from concurrent.futures import ThreadPoolExecutor

import torch
from loguru import logger
from packaging.version import parse
from tqdm import tqdm

from lightx2v.common.offload.block_slab import carve_block_slab, copy_block_slab_
from lightx2v.utils.profiler import ExcludedProfilingContext
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class WeightAsyncStreamManager(object):
    _NPU_SLOT_COUNT = 2

    def __init__(self, offload_granularity, load_stream=None, compute_stream=None):
        """Manage asynchronous weight copies and compute streams.

        ``load_stream`` and ``compute_stream`` are optional so multiple NPU
        managers can share the same pair of streams.  This is useful for models
        with more than one block stage, where creating independent streams for
        each stage makes the cross-stage dependency unnecessarily expensive.

        The existing CUDA/XPU APIs keep their synchronization behaviour.  NPU
        callers may opt in to the event-slot API (``prefetch_to_slot``,
        ``wait_ready`` and ``record_free``) to avoid host synchronization in
        the steady-state block loop.
        """
        self.offload_granularity = offload_granularity
        self.init_stream = torch_device_module.Stream(priority=0)
        self.need_init_first_buffer = True
        self.lazy_load = False
        torch_version = parse(torch.__version__.split("+")[0])
        if AI_DEVICE == "cuda" and torch_version >= parse("2.7"):
            load_stream_priority = 1
            compute_stream_priority = 1
        else:
            load_stream_priority = 0
            compute_stream_priority = -1

        self.cuda_load_stream = load_stream if load_stream is not None else torch_device_module.Stream(priority=load_stream_priority)
        self.compute_stream = compute_stream if compute_stream is not None else torch_device_module.Stream(priority=compute_stream_priority)

        if AI_DEVICE == "npu":
            self._npu_ready_events = [torch_device_module.Event() for _ in range(self._NPU_SLOT_COUNT)]
            self._npu_free_events = [torch_device_module.Event() for _ in range(self._NPU_SLOT_COUNT)]
            self._reset_npu_slot_state()

    def _reset_npu_slot_state(self):
        self._npu_slot_pending = [False] * self._NPU_SLOT_COUNT
        self._npu_slot_waited = [False] * self._NPU_SLOT_COUNT
        self._npu_free_recorded = [False] * self._NPU_SLOT_COUNT

    def _validate_npu_slot(self, slot_idx):
        if AI_DEVICE != "npu":
            raise RuntimeError("The event-slot offload API is only available on NPU")
        if self.offload_granularity != "block":
            raise RuntimeError("The event-slot offload API only supports block granularity")
        if not isinstance(slot_idx, int) or not 0 <= slot_idx < self._NPU_SLOT_COUNT:
            raise IndexError(f"slot_idx must be in [0, {self._NPU_SLOT_COUNT}), got {slot_idx!r}")
        if not hasattr(self, "cuda_buffers"):
            raise RuntimeError("init_cuda_buffer must be called before using an NPU event slot")
        if len(self.cuda_buffers) < self._NPU_SLOT_COUNT:
            raise RuntimeError(f"NPU event-slot offload requires {self._NPU_SLOT_COUNT} device buffers")

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
        """Enqueue one block copy into a fixed NPU staging slot.

        Reusing a slot inserts a device-side wait for its ``free`` event on the
        load stream.  Recording the corresponding ``ready`` event after the
        copy lets the compute stream consume the slot without blocking the host.
        """
        self._validate_npu_slot(slot_idx)
        if self._npu_slot_pending[slot_idx]:
            raise RuntimeError(f"NPU offload slot {slot_idx} is still pending; call record_free before reusing it")

        with torch_device_module.stream(self.cuda_load_stream):
            if self._npu_free_recorded[slot_idx]:
                self.cuda_load_stream.wait_event(self._npu_free_events[slot_idx])
            self._load_block_to_buffer(
                self.cuda_buffers[slot_idx],
                block_idx,
                blocks,
                adapter_block_idx,
            )
            self._npu_ready_events[slot_idx].record(self.cuda_load_stream)

        self._npu_slot_pending[slot_idx] = True
        self._npu_slot_waited[slot_idx] = False
        return self.cuda_buffers[slot_idx]

    def wait_ready(self, slot_idx, stream=None):
        """Make ``stream`` wait for the slot's ready event on the device."""
        self._validate_npu_slot(slot_idx)
        if not self._npu_slot_pending[slot_idx]:
            raise RuntimeError(f"NPU offload slot {slot_idx} has not been prefetched")

        stream = self.compute_stream if stream is None else stream
        stream.wait_event(self._npu_ready_events[slot_idx])
        self._npu_slot_waited[slot_idx] = True
        return self.cuda_buffers[slot_idx]

    def record_free(self, slot_idx, stream=None):
        """Record that compute has finished consuming an NPU staging slot."""
        self._validate_npu_slot(slot_idx)
        if not self._npu_slot_pending[slot_idx]:
            raise RuntimeError(f"NPU offload slot {slot_idx} has not been prefetched")
        if not self._npu_slot_waited[slot_idx]:
            raise RuntimeError(f"wait_ready must be called before record_free for NPU offload slot {slot_idx}")

        stream = self.compute_stream if stream is None else stream
        self._npu_free_events[slot_idx].record(stream)
        self._npu_slot_pending[slot_idx] = False
        self._npu_slot_waited[slot_idx] = False
        self._npu_free_recorded[slot_idx] = True

    def flush(self):
        """Drain NPU offload streams at a pipeline boundary and reset slots.

        This is intentionally the only event-slot method that synchronizes the
        host.  It should not be called inside the steady-state block loop.
        """
        if AI_DEVICE != "npu":
            raise RuntimeError("The event-slot offload API is only available on NPU")
        self.cuda_load_stream.synchronize()
        if self.compute_stream is not self.cuda_load_stream:
            self.compute_stream.synchronize()
        self._reset_npu_slot_state()

    def init_cpu_buffer(self, blocks_cpu_buffer=None, phases_cpu_buffer=None):
        self.need_init_first_buffer = True
        if self.offload_granularity == "block":
            assert blocks_cpu_buffer is not None
            self.cpu_buffers = [blocks_cpu_buffer[i] for i in range(len(blocks_cpu_buffer))]
        elif self.offload_granularity == "phase":
            assert phases_cpu_buffer is not None
            self.cpu_buffers = [phases_cpu_buffer[i] for i in range(len(phases_cpu_buffer))]
        else:
            raise NotImplementedError

    def init_cuda_buffer(self, blocks_cuda_buffer=None, phases_cuda_buffer=None):
        self.need_init_first_buffer = True
        if self.offload_granularity == "block":
            assert blocks_cuda_buffer is not None
            self.cuda_buffers = [blocks_cuda_buffer[i] for i in range(len(blocks_cuda_buffer))]
        elif self.offload_granularity == "phase":
            assert phases_cuda_buffer is not None
            self.cuda_buffers = [phases_cuda_buffer[i] for i in range(len(phases_cuda_buffer))]
        else:
            raise NotImplementedError

    def init_block_slabs(self, block_slabs, staging_raw=None):
        """Initialize the one-H2D-per-block fast path.

        All typed views alias one largest-block device allocation.  The load
        stream copies a CPU slab into that allocation once, then the existing
        weight ``load_state_dict`` implementations scatter the typed device
        views into their established compute buffers.
        """
        if self.offload_granularity != "block":
            raise ValueError("block slabs require block offload granularity")

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

    def _sync(self):
        """Synchronize to ensure memory visibility across streams.

        XPU streams do not guarantee cross-stream memory visibility after a
        per-stream synchronize, so we use a device-wide sync on XPU.
        On CUDA, per-stream synchronize is sufficient and preferred.
        """
        if AI_DEVICE == "xpu":
            torch_device_module.synchronize()
        else:
            self.init_stream.synchronize()

    def init_first_buffer(self, blocks, adapter_block_idx=None):
        with torch_device_module.stream(self.init_stream):
            if hasattr(self, "cpu_buffers"):
                if self.offload_granularity == "block":
                    self.cuda_buffers[0].load_state_dict(self.cpu_buffers[0].state_dict(), 0, adapter_block_idx)
                else:
                    self.cuda_buffers[0].load_state_dict(self.cpu_buffers[0][0].state_dict(), 0, adapter_block_idx)
            else:
                if self.offload_granularity == "block":
                    self.cuda_buffers[0].load_state_dict(blocks[0].state_dict(), 0, adapter_block_idx)
                else:
                    self.cuda_buffers[0].load_state_dict(blocks[0].compute_phases[0].state_dict(), 0, adapter_block_idx)
        self._sync()
        self.need_init_first_buffer = False

    def prefetch_weights(self, block_idx, blocks, adapter_block_idx=None):
        with torch_device_module.stream(self.cuda_load_stream):
            if hasattr(self, "cpu_buffers"):
                self.cuda_buffers[1].load_state_dict(self.cpu_buffers[0].state_dict(), block_idx, adapter_block_idx)
            else:
                self.cuda_buffers[1].load_state_dict(blocks[block_idx].state_dict(), block_idx, adapter_block_idx)

    def prefetch_phase(self, block_idx, phase_idx, blocks, adapter_block_idx=None):
        with torch_device_module.stream(self.cuda_load_stream):
            if hasattr(self, "cpu_buffers"):
                self.cuda_buffers[phase_idx].load_state_dict(self.cpu_buffers[0][phase_idx].state_dict(), block_idx, adapter_block_idx)
            else:
                self.cuda_buffers[phase_idx].load_state_dict(blocks[block_idx].compute_phases[phase_idx].state_dict(), block_idx, adapter_block_idx)

    def swap_blocks(self):
        if AI_DEVICE == "xpu":
            torch_device_module.synchronize()
        else:
            self.cuda_load_stream.synchronize()
            self.compute_stream.synchronize()
        self.cuda_buffers[0], self.cuda_buffers[1] = (
            self.cuda_buffers[1],
            self.cuda_buffers[0],
        )

    def swap_phases(self):
        if AI_DEVICE == "xpu":
            torch_device_module.synchronize()
        else:
            self.cuda_load_stream.synchronize()
            self.compute_stream.synchronize()

    @ExcludedProfilingContext("🔥 warm_up_cpu_buffers")
    def warm_up_cpu_buffers(self, blocks_num):
        logger.info("🔥 Warming up cpu buffers...")
        for i in tqdm(range(blocks_num)):
            for phase in self.cpu_buffers[0]:
                phase.load_state_dict_from_disk(i, None)
            for phase in self.cpu_buffers[1]:
                phase.load_state_dict_from_disk(i, None)

        for phase in self.cpu_buffers[0]:
            phase.load_state_dict_from_disk(0, None)
        for phase in self.cpu_buffers[1]:
            phase.load_state_dict_from_disk(1, None)
        logger.info("✅ CPU buffers warm-up completed.")

    def init_lazy_load(self, num_workers=6):
        self.lazy_load = True
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.prefetch_futures = []
        self.prefetch_block_idx = -1

    def start_prefetch_block(self, block_idx, adapter_block_idx=None):
        self.prefetch_block_idx = block_idx
        self.prefetch_futures = []
        if self.offload_granularity == "block":
            future = self.executor.submit(self.cpu_buffers[1].load_state_dict_from_disk, block_idx, adapter_block_idx)
            self.prefetch_futures.append(future)
        else:
            for phase in self.cpu_buffers[1]:
                future = self.executor.submit(phase.load_state_dict_from_disk, block_idx, adapter_block_idx)
                self.prefetch_futures.append(future)

    def swap_cpu_buffers(self):
        # import time
        # wait_start = time.time()
        # already_done = all(f.done() for f in self.prefetch_futures)
        for f in self.prefetch_futures:
            f.result()
        # wait_time = time.time() - wait_start
        # logger.debug(f"[Prefetch] block {self.prefetch_block_idx}: wait={wait_time:.3f}s, already_done={already_done}")
        self.cpu_buffers = [self.cpu_buffers[1], self.cpu_buffers[0]]

    def __del__(self):
        if hasattr(self, "executor") and self.executor is not None:
            for f in self.prefetch_futures:
                if not f.done():
                    f.result()
            self.executor.shutdown(wait=False)
            self.executor = None
            logger.debug("ThreadPoolExecutor shut down successfully.")
