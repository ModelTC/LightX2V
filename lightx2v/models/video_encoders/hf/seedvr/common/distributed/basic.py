

"""
Distributed basic functions.
"""

import os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def get_global_rank() -> int:
    """
    Get the global rank, the global index of the GPU.
    """
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """
    Get the local rank, the local index of the GPU.
    """
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """
    Get the world size, the total amount of GPUs.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_device() -> torch.device:
    """
    Get current rank device.
    """
    return torch.device("cuda", get_local_rank())


def barrier_if_distributed(*args, **kwargs):
    """
    Synchronizes all processes if under distributed context.
    """
    if dist.is_initialized():
        return dist.barrier(*args, **kwargs)


def init_torch(cudnn_benchmark=True, timeout=timedelta(seconds=600)):
    """
    Common PyTorch initialization configuration.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.cuda.set_device(get_local_rank())
    dist.init_process_group(
        backend="nccl",
        rank=get_global_rank(),
        world_size=get_world_size(),
        timeout=timeout,
    )


def convert_to_ddp(module: torch.nn.Module, **kwargs) -> DistributedDataParallel:
    return DistributedDataParallel(
        module=module,
        device_ids=[get_local_rank()],
        output_device=get_local_rank(),
        **kwargs,
    )
