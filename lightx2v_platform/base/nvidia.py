import os

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
        if ProcessGroupNCCL is None:
            raise RuntimeError("ProcessGroupNCCL is not available. Please check your runtime environment.")
        device_count = torch.cuda.device_count()
        device_index_value = os.getenv("LIGHTX2V_CUDA_DEVICE_INDEX", os.getenv("LOCAL_RANK"))
        if device_index_value is None:
            device_index = int(os.getenv("RANK", "0")) % max(device_count, 1)
        else:
            device_index = int(device_index_value)
        if device_index < 0 or device_index >= device_count:
            raise RuntimeError(
                f"Distributed CUDA device index {device_index} is outside visible device count {device_count}."
            )
        torch.cuda.set_device(device_index)
        pg_options = ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = True
        dist.init_process_group(backend="nccl", pg_options=pg_options)
