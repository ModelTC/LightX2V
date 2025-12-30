import torch
import torch.distributed as dist

from lightx2v_platform.base.nvidia import CudaDevice
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


@PLATFORM_DEVICE_REGISTER("metax_cuda")
class MetaxDevice(CudaDevice):
    name = "cuda"

    @staticmethod
    def init_device_env():
        pass

    @staticmethod
    def is_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def get_device() -> str:
        return "cuda"

    @staticmethod
    def init_parallel_env():
        dist.init_process_group(backend="nccl")
        torch.npu.set_device(dist.get_rank())
