# Minimal distributed utilities extracted from lyra_2._ext.imaginaire.utils.distributed
# Only the functions used in the inference path are implemented.

import torch
import torch.distributed as dist


def get_rank(group=None) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group)
    return 0


def is_rank0() -> bool:
    return get_rank() == 0


def broadcast(tensor: torch.Tensor, src: int = 0, group=None) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(tensor, src=src, group=group)
