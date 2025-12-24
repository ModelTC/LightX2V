import torch
import torch.distributed as dist

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


@PLATFORM_DEVICE_REGISTER("npu")
class NpuDevice:
    name = "npu"

    @staticmethod
    def is_available() -> bool:
        try:
            import torch_npu

            assert torch_npu

            return torch.npu.is_available()
        except ImportError:
            return False

    @staticmethod
    def get_device() -> str:
        return "npu"

    @staticmethod
    def init_parallel_env():
        dist.init_process_group(backend="hccl")
        torch.npu.set_device(dist.get_rank())
