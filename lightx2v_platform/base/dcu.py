import torch
import torch.distributed as dist

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


@PLATFORM_DEVICE_REGISTER("dcu")
class DcuDevice:
    """
    DCU (AMD GPU) Device implementation for LightX2V.

    DCU uses ROCm which provides CUDA-compatible APIs through HIP.
    Most PyTorch operations work transparently through the ROCm backend.
    """

    name = "dcu"

    @staticmethod
    def is_available() -> bool:
        """
        Check if DCU is available.

        DCU uses the standard CUDA API through ROCm's HIP compatibility layer.
        Returns:
            bool: True if DCU/CUDA is available
        """
        try:
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def get_device() -> str:
        """
        Get the device type string.

        Returns "cuda" because DCU uses CUDA-compatible APIs through ROCm.
        This allows seamless integration with existing PyTorch code.

        Returns:
            str: "cuda" for ROCm compatibility
        """
        return "cuda"

    @staticmethod
    def init_parallel_env():
        """
        Initialize distributed parallel environment for DCU.

        Uses RCCL (ROCm Collective Communications Library) which is
        compatible with NCCL APIs for multi-GPU communication.
        """
        # RCCL is compatible with NCCL backend
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
