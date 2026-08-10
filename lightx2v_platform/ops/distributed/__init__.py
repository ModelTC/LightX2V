"""Default distributed collective implementations."""

import torch.distributed as dist


def tensor_parallel_reduce(tensor, group, world_size, prefer_all_gather=False):
    """Reduce tensor-parallel partials in place.

    Platforms may override this operation when their communication runtime has
    a more suitable collective for a given execution topology.
    """
    del world_size, prefer_all_gather
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor
