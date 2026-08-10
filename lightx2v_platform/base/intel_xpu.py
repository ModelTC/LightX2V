"""
Intel XPU Device implementation for LightX2V.

Intel XPU provides GPU acceleration through Intel's hardware (Arc GPU, Ponte Vecchio, etc.).
This module handles Intel-specific configurations including:
- XPU device initialization
- Distributed training with Intel oneCCL backend
"""

import os
import platform

import torch
import torch.distributed as dist
from loguru import logger
from packaging.version import Version

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

# Detect Intel XPU platform
IS_INTEL_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


@PLATFORM_DEVICE_REGISTER("intel_xpu")
class IntelXpuDevice:
    """
    Intel XPU Device implementation for LightX2V.

    Intel XPU uses torch.xpu APIs for GPU acceleration.
    Distributed training uses Intel oneCCL backend.
    """

    name = "intel_xpu"

    @staticmethod
    def init_device_env():
        """
        Initialize Intel XPU optimizations.

        This is called from lightx2v_platform.set_ai_device when platform is intel_xpu.
        Currently no specific optimizations needed for Intel XPU.
        """
        logger.info("Intel XPU platform detected, initializing environment...")
        logger.info(f"  - Available XPU devices: {torch.xpu.device_count()}")

    @staticmethod
    def is_available() -> bool:
        """Check if Intel XPU is available."""
        return IS_INTEL_XPU

    @staticmethod
    def get_device() -> str:
        """Get the device type string. Returns 'xpu' for Intel XPU."""
        return "xpu"

    @staticmethod
    def init_parallel_env():
        """Initialize a single-node distributed environment for Intel XPU."""
        if platform.system() != "Linux" and Version(torch.__version__) < Version("2.10.0+xpu"):
            raise RuntimeError(f"Intel XPU distributed initialization on non-Linux systems requires PyTorch >= 2.10.0+xpu. Found PyTorch {torch.__version__}.")

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.xpu.set_device(local_rank)
        dist.init_process_group(backend="xccl")


# Register alias "xpu" for backward compatibility
PLATFORM_DEVICE_REGISTER["xpu"] = IntelXpuDevice
