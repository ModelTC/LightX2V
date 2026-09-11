"""Double-buffered disk offload into CPU-visible MPS weight storage."""

from contextlib import nullcontext

import torch
from loguru import logger

from lightx2v.common.offload.manager import WeightAsyncStreamManager


def host_view(tensor):
    """Alias MPS storage without copying; the caller orders CPU/GPU accesses."""
    if tensor.device.type != "mps":
        raise ValueError("Shared weight buffers require an MPS tensor")
    if not hasattr(torch.mps, "_host_alias_storage"):
        raise RuntimeError("Shared MPS weight buffers require torch.mps._host_alias_storage (PyTorch >= 2.13)")
    storage = torch.mps._host_alias_storage(tensor.untyped_storage())
    return torch.empty(0, dtype=tensor.dtype).set_(storage, tensor.storage_offset(), tensor.shape, tensor.stride())


class MpsSharedWeightAsyncStreamManager(WeightAsyncStreamManager):
    """Read into the idle slot while the default MPS stream uses the other.

    The source implements load_block_into(target_buffer, block_idx). This runs
    on a CPU worker and must only write host views of the supplied idle buffer.
    """

    def __init__(self, offload_granularity="block"):
        if offload_granularity != "block":
            raise ValueError("Shared MPS weight offload only supports block granularity")
        super().__init__(offload_granularity)
        self.cuda_buffers = []
        self.init_lazy_load(num_workers=1)
        logger.info("MPS shared weight offload: two device buffers, direct file reads, one prefetch worker")

    def _init_streams(self):
        # Disk I/O runs on a CPU thread; GPU work stays on the default stream.
        pass

    def prepare_compute(self):
        pass

    def compute_context(self):
        return nullcontext()

    def init_cuda_buffer(self, blocks_cuda_buffer=None, phases_cuda_buffer=None):
        if blocks_cuda_buffer is None or len(blocks_cuda_buffer) != 2:
            raise ValueError("Shared MPS offload requires exactly two device buffers")
        if self.prefetch_futures:
            raise RuntimeError("Cannot replace shared buffers while a prefetch is pending")
        super().init_cuda_buffer(blocks_cuda_buffer, phases_cuda_buffer)
        if self.executor is None:
            self.init_lazy_load(num_workers=1)

    def init_first_buffer(self, blocks, adapter_block_idx=None):
        torch.mps.synchronize()
        blocks.load_block_into(self.cuda_buffers[0], 0)
        torch.mps.synchronize()
        self.need_init_first_buffer = False

    def prefetch_weights(self, block_idx, blocks, adapter_block_idx=None):
        if self.prefetch_futures:
            raise RuntimeError("Call swap_blocks before scheduling another prefetch")
        self.prefetch_block_idx = block_idx
        self.prefetch_futures = [self.executor.submit(blocks.load_block_into, self.cuda_buffers[1], block_idx)]

    def swap_blocks(self):
        if not self.prefetch_futures:
            raise RuntimeError("No shared-buffer prefetch to complete")
        # This commits GPU work and releases the GIL while waiting, so the disk
        # reader continues filling slot 1 while the GPU consumes slot 0.
        torch.mps.synchronize()
        for future in self.prefetch_futures:
            future.result()
        torch.mps.synchronize()
        self.prefetch_futures.clear()
        self.cuda_buffers.reverse()

    def close(self):
        """Drain both users before releasing aliases; allow later reinitialization."""
        try:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
            torch.mps.synchronize()
        finally:
            self.executor = None
            self.prefetch_futures.clear()
            self.cuda_buffers = []
            self.need_init_first_buffer = True
