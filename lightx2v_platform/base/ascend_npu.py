import torch
import torch.distributed as dist

from lightx2v_platform.base.offload import TorchBlockOffload
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


class NpuBlockOffload(TorchBlockOffload):
    validate_checkpoint = True

    @staticmethod
    def prepare():
        # Byte views require ND storage, before any block buffers are created.
        torch.npu.config.allow_internal_format = False


@PLATFORM_DEVICE_REGISTER("ascend_npu")
class NpuDevice:
    name = "ascend_npu"
    block_offload_backend = NpuBlockOffload

    @staticmethod
    def init_device_env():
        import lightx2v_platform.compilation.ascend_npu  # noqa: F401

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
    def copy_to_cpu(destination, source, non_blocking=False):
        """Copy to CPU without losing the strides of an existing host tensor."""
        if not destination.is_contiguous():
            # NPU D2H copy_ can ignore a strided host destination's layout.
            # Complete D2H before the CPU copy, even for a non-blocking request.
            return destination.copy_(source.contiguous().to("cpu", non_blocking=False))
        return destination.copy_(source, non_blocking=non_blocking)

    @staticmethod
    def init_parallel_env():
        dist.init_process_group(backend="hccl")
        torch.npu.set_device(dist.get_rank())
