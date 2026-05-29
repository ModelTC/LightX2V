import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

_DEVICE_MESH = None


def init_distributed(config=None):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist_config = (config or {}).get("distributed", {})
        backend = dist_config.get("backend", "nccl")
        timeout_minutes = dist_config.get("timeout_minutes", 10)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))

    global _DEVICE_MESH
    if _DEVICE_MESH is None:
        _DEVICE_MESH = init_device_mesh("cuda", (dist.get_world_size(),))


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_main_process():
    return get_rank() == 0


def get_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda", torch.cuda.current_device())


def get_device_mesh():
    return _DEVICE_MESH


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def rank_zero_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def reduce_mean(value):
    if not is_distributed():
        return value
    tensor = torch.as_tensor(value, device=get_device(), dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor.item()
