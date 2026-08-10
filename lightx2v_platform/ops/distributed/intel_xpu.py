"""Intel XPU distributed collective implementations."""

import torch
import torch.distributed as dist


def tensor_parallel_reduce(tensor, group, world_size, prefer_all_gather=False):
    """Reduce TP partials without overlapping oneCCL all-reduce with SP A2A."""
    if not prefer_all_gather:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    partials = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(partials, tensor.contiguous(), group=group)
    tensor.copy_(partials[0])
    for partial in partials[1:]:
        tensor.add_(partial)
    return tensor
