"""P2P communication manager for pipeline parallelism.

Uses the default CUDA stream for all NCCL operations. NCCL's internal
stream handles the actual async data transfer, so a separate comm_stream
is unnecessary and would only introduce cross-stream sync overhead.

isend requests are returned to the caller who MUST store them to
prevent tensor GC before the send completes.
"""

from typing import Dict, List, Tuple

import torch
import torch.distributed as dist


class PipelineComm:
    """P2P communication between adjacent pipeline stages.

    CUDA-only: receive buffers are allocated on the current CUDA device via
    ``torch.cuda.current_device()``. PipeFusion is gated to CUDA at config time
    (see ``set_config._validate_pipefusion_config``), so this is not reachable
    on XPU/NPU platforms.
    """

    def __init__(self, pp_group: dist.ProcessGroup):
        self.pp_group = pp_group
        self.rank = dist.get_rank(pp_group)
        self.world_size = dist.get_world_size(pp_group)

        self.ranks = list(dist.get_process_group_ranks(pp_group))
        self.prev_rank = self.ranks[(self.rank - 1) % self.world_size]
        self.next_rank = self.ranks[(self.rank + 1) % self.world_size]

        self._recv_tasks_queue: List[Tuple[str, int]] = []
        self._receiving_tasks: List[Tuple[object, str, int]] = []
        self._recv_buffers: Dict[Tuple[str, int], torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Synchronous send / recv (sync pipeline)
    # ------------------------------------------------------------------

    def pipeline_send(self, tensor: torch.Tensor):
        dist.send(tensor.contiguous(), dst=self.next_rank, group=self.pp_group)

    def pipeline_recv(self, shape, dtype=None) -> torch.Tensor:
        if dtype is None:
            dtype = torch.bfloat16
        buf = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
        dist.recv(buf, src=self.prev_rank, group=self.pp_group)
        return buf

    # ------------------------------------------------------------------
    # Asynchronous send / recv (async pipeline)
    # All on default stream — NCCL's internal stream handles the real
    # async transfer, and req.wait() just inserts a stream-side
    # dependency (non-blocking on CPU in the same-stream case).
    # ------------------------------------------------------------------

    def pipeline_isend(self, tensor: torch.Tensor, name: str = "latent", segment_idx: int = 0):
        """Non-blocking send on the current (default) stream.

        Returns ``(Work, actual_send_tensor)``. ``actual_send_tensor`` is the
        contiguous buffer actually handed to NCCL (``contiguous()`` may copy),
        so the caller must keep BOTH alive until ``Work.wait()`` completes.
        The Work's wait() only inserts a stream-side dependency — it does not
        block the CPU thread.
        """
        tensor = tensor.contiguous()
        return dist.isend(tensor, dst=self.next_rank, group=self.pp_group), tensor

    def add_pipeline_recv_task(self, idx: int = 0, name: str = "latent", shape=None, dtype=None):
        self._recv_tasks_queue.append((name, idx))
        if (name, idx) not in self._recv_buffers:
            assert shape is not None and dtype is not None
            self._recv_buffers[(name, idx)] = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())

    def recv_next(self):
        """Post next irecv on the current (default) stream.

        Non-blocking on CPU: dist.irecv enqueues the work on NCCL's
        internal stream and returns immediately.
        """
        if not self._recv_tasks_queue:
            raise ValueError("No more tasks to receive")
        name, idx = self._recv_tasks_queue.pop(0)
        buf = self._recv_buffers.get((name, idx))
        assert buf is not None
        req = dist.irecv(buf, src=self.prev_rank, group=self.pp_group)
        self._receiving_tasks.append((req, name, idx))

    def get_pipeline_recv_data(self, idx: int = 0, name: str = "latent") -> torch.Tensor:
        """Wait for and return a previously posted async receive.

        In the single-stream model, req.wait() inserts a stream-side
        wait for the NCCL op's completion on the current stream. It
        does NOT block the CPU thread unless a timeout is set.
        """
        assert self._receiving_tasks
        req, rname, ridx = self._receiving_tasks.pop(0)
        assert rname == name and ridx == idx
        req.wait()
        return self._recv_buffers[(name, idx)]
