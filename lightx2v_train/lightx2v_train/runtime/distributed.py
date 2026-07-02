import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

_DEVICE_MESH = None
_FSDP_DEVICE_MESH = None
_DP_GROUP = None
_SP_GROUP = None
_DP_RANK = 0
_DP_WORLD_SIZE = 1
_SP_RANK = 0
_SP_WORLD_SIZE = 1


def _positive_int(value, name):
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _get_int_config(config, paths):
    for path in paths:
        current = config or {}
        found = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                found = False
                break
            current = current[key]
        if found and current is not None:
            return current
    return None


def _resolve_sequence_parallel_size(config):
    dist_config = (config or {}).get("distributed", {})
    sp_config = dist_config.get("sequence_parallel")
    if isinstance(sp_config, dict):
        if not sp_config.get("enabled", False):
            return 1
        if "size" not in sp_config:
            raise ValueError("distributed.sequence_parallel.size is required when sequence_parallel.enabled=true.")
        return _positive_int(sp_config["size"], "distributed.sequence_parallel.size")

    sp_size = _get_int_config(
        config,
        (
            ("distributed", "sp_size"),
            ("distributed", "sequence_parallel_size"),
        ),
    )
    if sp_size is not None:
        return _positive_int(sp_size, "distributed.sp_size")

    sp_config = dist_config.get("sp", {})
    if isinstance(sp_config, int):
        return _positive_int(sp_config, "distributed.sequence_parallel")
    if isinstance(sp_config, dict):
        if not sp_config.get("enabled", True):
            return 1
        if "size" in sp_config:
            return _positive_int(sp_config["size"], "distributed.sequence_parallel.size")
    model_config = (config or {}).get("model", {})
    return _positive_int(model_config.get("sequence_parallel_size", 1), "model.sequence_parallel_size")


def _resolve_fsdp_size(config):
    dist_config = (config or {}).get("distributed", {})
    fsdp_config = dist_config.get("fsdp2")
    if isinstance(fsdp_config, dict):
        if "size" in fsdp_config:
            return _positive_int(fsdp_config["size"], "distributed.fsdp2.size")
        if not fsdp_config.get("enabled", False):
            return 1

    fsdp_size = _get_int_config(
        config,
        (
            ("distributed", "fsdp_size"),
            ("distributed", "dp_size"),
            ("distributed", "data_parallel_size"),
            ("distributed", "fsdp2", "fsdp_size"),
        ),
    )
    if fsdp_size is None:
        return None
    return _positive_int(fsdp_size, "distributed.fsdp_size")


def _resolve_parallel_sizes(config, world_size):
    sp_size = _resolve_sequence_parallel_size(config)
    fsdp_size = _resolve_fsdp_size(config)
    if fsdp_size is None:
        if world_size % sp_size != 0:
            raise ValueError(f"distributed.sequence_parallel.size={sp_size} must divide WORLD_SIZE={world_size}.")
        fsdp_size = world_size // sp_size

    if sp_size * fsdp_size != world_size:
        raise ValueError(f"distributed.sequence_parallel.size * distributed.fsdp2.size must equal WORLD_SIZE: {sp_size} * {fsdp_size} != {world_size}.")
    return sp_size, fsdp_size


def init_distributed(config=None):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    dist_config = (config or {}).get("distributed", {})
    backend = dist_config.get("backend", "nccl")
    if not dist.is_initialized():
        timeout_minutes = dist_config.get("timeout_minutes", 10)
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))

    global _DEVICE_MESH, _FSDP_DEVICE_MESH
    global _DP_GROUP, _SP_GROUP, _DP_RANK, _DP_WORLD_SIZE, _SP_RANK, _SP_WORLD_SIZE
    if _DEVICE_MESH is None:
        world_size = dist.get_world_size()
        sp_size, fsdp_size = _resolve_parallel_sizes(config, world_size)
        dp_size = fsdp_size

        _SP_WORLD_SIZE = sp_size
        _DP_WORLD_SIZE = dp_size
        global_rank = dist.get_rank()
        _DP_RANK = global_rank // sp_size
        _SP_RANK = global_rank % sp_size

        if sp_size > 1:
            _DEVICE_MESH = init_device_mesh(
                "cuda",
                (dp_size, sp_size),
                mesh_dim_names=("dp", "sp"),
            )
            _FSDP_DEVICE_MESH = _DEVICE_MESH["dp"]
            _DP_GROUP = _DEVICE_MESH["dp"].get_group()
            _SP_GROUP = _DEVICE_MESH["sp"].get_group()
        else:
            _DEVICE_MESH = init_device_mesh("cuda", (world_size,))
            _FSDP_DEVICE_MESH = _DEVICE_MESH
            _DP_GROUP = _DEVICE_MESH.get_group()
            _SP_GROUP = None


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def get_data_parallel_rank():
    return _DP_RANK if is_distributed() else 0


def get_data_parallel_world_size():
    return _DP_WORLD_SIZE if is_distributed() else 1


def get_data_parallel_group():
    return _DP_GROUP


def get_sequence_parallel_rank():
    return _SP_RANK if is_distributed() else 0


def get_sequence_parallel_world_size():
    return _SP_WORLD_SIZE if is_distributed() else 1


def get_sequence_parallel_group():
    return _SP_GROUP


def is_sequence_parallel_enabled():
    return is_distributed() and _SP_WORLD_SIZE > 1


def get_sequence_parallel_src_rank(src_sp_rank=0):
    if not is_sequence_parallel_enabled():
        return get_rank()
    return get_data_parallel_rank() * get_sequence_parallel_world_size() + int(src_sp_rank)


def is_main_process():
    return get_rank() == 0


def get_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda", torch.cuda.current_device())


def get_device_mesh():
    return _FSDP_DEVICE_MESH


def get_parallel_device_mesh():
    return _DEVICE_MESH


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_mean(value):
    if not is_distributed():
        return value
    tensor = torch.as_tensor(value, device=get_device(), dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor.item()
