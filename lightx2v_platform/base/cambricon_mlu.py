import torch
import torch.distributed as dist

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


@PLATFORM_DEVICE_REGISTER("mlu")
class MluDevice:
    name = "mlu"

    @staticmethod
    def init_device_env():
        pass

    @staticmethod
    def is_available() -> bool:
        try:
            import torch_mlu

            return torch_mlu.mlu.is_available()
        except ImportError:
            return False

    @staticmethod
    def get_device() -> str:
        return "mlu"

    @staticmethod
    def init_parallel_env():
        dist.init_process_group(backend="cncl")
        torch.mlu.set_device(dist.get_rank())
