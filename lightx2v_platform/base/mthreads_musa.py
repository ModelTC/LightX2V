import os

import torch
import torch.distributed as dist

from lightx2v_platform.base.nvidia import CudaDevice
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


@PLATFORM_DEVICE_REGISTER("musa")
class MusaDevice(CudaDevice):
    name = "musa"

    @staticmethod
    def init_parallel_env():
        if not hasattr(torch, "musa") or not torch.musa.is_available():
            raise RuntimeError("MUSA is not available. Check the MThreads device mappings and runtime.")
        dist.init_process_group(backend="mccl")
        local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank() % torch.musa.device_count()))
        torch.musa.set_device(local_rank)

    @staticmethod
    def is_available() -> bool:
        try:
            import torch
            import torchada  # noqa: F401

            return hasattr(torch, "musa") and torch.musa.is_available()
        except ImportError:
            return False
