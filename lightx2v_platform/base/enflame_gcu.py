# Import torch_gcu BEFORE torch.distributed to enable ECCL/NCCL backend
# This is critical - ECCL needs to register NCCL backend before PyTorch's
# distributed module checks for it
try:
    import torch_gcu  # noqa: F401
    from torch_gcu import transfer_to_gcu  # noqa: F401
except ImportError:
    pass  # torch_gcu not available, will be handled in is_available()

import torch
import torch.distributed as dist

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


@PLATFORM_DEVICE_REGISTER("gcu")
class EnflameGcuDevice:
    """
    Enflame GCU Device implementation for LightX2V.

    Enflame GCU uses torch_gcu which provides CUDA-compatible APIs.
    Most PyTorch operations work transparently through the GCU backend.
    """

    name = "gcu"

    @staticmethod
    def is_available() -> bool:
        """
        Check if Enflame GCU is available.

        Uses torch_gcu.gcu.is_available() to check device availability.

        Returns:
            bool: True if Enflame GCU is available
        """
        try:
            import torch_gcu
            return torch_gcu.gcu.is_available()
        except ImportError:
            return False

    @staticmethod
    def get_device() -> str:
        """
        Get the device type string.

        Returns "gcu" for Enflame GCU device. This allows getattr(torch, AI_DEVICE)
        to work correctly (torch.gcu) and torch.device(AI_DEVICE) to work with GCU.

        Returns:
            str: "gcu" for GCU device
        """
        return "gcu"

    @staticmethod
    def init_parallel_env():
        """
        Initialize distributed parallel environment for Enflame GCU.

        Uses ECCL (Enflame Collective Communication Library) which is
        compatible with NCCL APIs for multi-GPU communication.
        """
        try:
            import torch_gcu
        except ImportError:
            raise ImportError("torch_gcu is not available. Please install torch_gcu for Enflame GCU support.")
        
        # Use NCCL backend directly (ECCL is compatible with NCCL API)
        # ECCL should make NCCL backend available through torch_gcu
        dist.init_process_group(backend="nccl")
        
        # Use torch.gcu.set_device() instead of torch.cuda.set_device()
        if hasattr(torch, 'gcu') and hasattr(torch.gcu, 'set_device'):
            torch.gcu.set_device(dist.get_rank())
        elif hasattr(torch_gcu, 'gcu') and hasattr(torch_gcu.gcu, 'set_device'):
            torch_gcu.gcu.set_device(dist.get_rank())
        else:
            # Fallback: try cuda API (may work if torch_gcu provides CUDA compatibility)
            try:
                torch.cuda.set_device(dist.get_rank())
            except Exception:
                # If all else fails, just log a warning
                import warnings
                warnings.warn("Could not set GCU device. Continuing without explicit device setting.")

