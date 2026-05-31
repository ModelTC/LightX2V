import torch
from loguru import logger
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

from lightx2v_train.runtime.distributed import get_device_mesh, is_distributed
from lightx2v_train.utils.utils import get_running_dtype


def fsdp2_enabled(config):
    fsdp_config = config.get("distributed", {}).get("fsdp2", {})
    return is_distributed() and fsdp_config.get("enabled", True)


def is_fsdp2_module(module):
    return isinstance(module, FSDPModule)


def _dtype(name):
    if name is None:
        return None
    if isinstance(name, torch.dtype):
        return name
    return get_running_dtype(name)


def _build_mp_policy(mp_config):
    return MixedPrecisionPolicy(
        param_dtype=_dtype(mp_config.get("param_dtype")),
        reduce_dtype=_dtype(mp_config.get("reduce_dtype")),
        output_dtype=_dtype(mp_config.get("output_dtype")),
        cast_forward_inputs=mp_config.get("cast_forward_inputs", False),
    )


def _fully_shard_module(module, mesh, mp_policy, reshard_after_forward):
    return fully_shard(
        module,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )


def _cuda_memory_gb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3


def _iter_shard_plan(plan):
    for entry in plan:
        reshard_after_forward = entry["reshard_after_forward"]
        if "module" in entry:
            yield entry["module"], reshard_after_forward
        else:
            for module in entry["modules"]:
                yield module, reshard_after_forward


def apply_fsdp2(model, config):
    if not fsdp2_enabled(config) or model.is_fsdp2_wrapped():
        return model

    fsdp_config = config.get("distributed", {}).get("fsdp2", {})
    before = _cuda_memory_gb()
    mp_config = fsdp_config.get("mixed_precision", {})
    mp_policy = _build_mp_policy(mp_config)
    mesh = get_device_mesh()

    for module, reshard_after_forward in _iter_shard_plan(model.fsdp2_shard_plan(fsdp_config)):
        _fully_shard_module(module, mesh, mp_policy, reshard_after_forward)

    torch.cuda.empty_cache()

    if fsdp_config.get("log_memory", True):
        after = _cuda_memory_gb()
        if before is not None and after is not None:
            logger.info(
                "FSDP2 transformer sharded: allocated {:.2f} -> {:.2f} GiB, reserved {:.2f} -> {:.2f} GiB",
                before[0],
                after[0],
                before[1],
                after[1],
            )
    return model
