import torch
import torch.distributed as dist

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

try:
    from torch.distributed import ProcessGroupNCCL
except ImportError:
    ProcessGroupNCCL = None


@PLATFORM_DEVICE_REGISTER("cuda")
class CudaDevice:
    name = "cuda"

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
        if ProcessGroupNCCL is None:
            raise RuntimeError("ProcessGroupNCCL is not available. Please check your runtime environment.")
        pg_options = ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = True
        dist.init_process_group(backend="nccl", pg_options=pg_options)
        torch.cuda.set_device(dist.get_rank())
