"""Exchange the temporal boundary between consecutive SwiftVR chunks."""

import torch

from lightx2v.common.ops.attn.utils.ring_comm import RingComm


def exchange_chunk_boundary(tensor: torch.Tensor, group) -> torch.Tensor:
    comm = RingComm(group)
    received = comm.enqueue_send_recv(tensor)
    comm.commit()
    comm.wait()
    return received
