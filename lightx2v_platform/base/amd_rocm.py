"""
AMD ROCm Device implementation for LightX2V.

AMD ROCm provides CUDA-compatible APIs through HIP (Heterogeneous-computing Interface for Portability).
This module handles AMD-specific optimizations including:
- Disabling cudnn for faster VAE convolution
- sgl_kernel compatibility layer using aiter library (required on AMD)
"""

import sys
import torch
import torch.distributed as dist

from loguru import logger

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

# Detect AMD ROCm platform
IS_AMD_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None

# aiter installation info
AITER_REPO = "https://github.com/ROCm/aiter.git"
AITER_COMMIT = "a7d3bf8cd47afbaf6a6133c1f12e3b01d2c27b0e"
AITER_INSTALL_CMD = f"""
# One-line install command for aiter (AMD ROCm optimized kernels):
git clone {AITER_REPO} /tmp/aiter && \\
cd /tmp/aiter && \\
git checkout {AITER_COMMIT} && \\
pip install -e .
"""


class AiterSglKernelCompat:
    """
    Compatibility layer to use aiter with sgl_kernel interface.
    
    This class wraps aiter functions to match sgl_kernel's API,
    allowing existing code to work seamlessly on AMD GPUs.
    
    Note: This is REQUIRED on AMD ROCm as the original sgl_kernel
    does not support AMD GPUs.
    """
    
    def __init__(self, aiter_module):
        self._aiter = aiter_module
        self._gemm_a8w8 = aiter_module.gemm_a8w8_CK
        self._pertoken_quant = aiter_module.pertoken_quant
        self._dtypes = aiter_module.dtypes
        self._rms_norm = aiter_module.rms_norm
        logger.info("Using aiter as sgl_kernel backend (AMD ROCm optimized)")
    
    def rmsnorm(self, input, weight, eps):
        """RMSNorm compatible with sgl_kernel.rmsnorm(input, weight, eps)"""
        return self._rms_norm(input, weight, eps)
    
    def fp8_scaled_mm(self, input_quant, weight, input_scale, weight_scale, dtype, bias=None):
        """FP8 GEMM compatible with sgl_kernel.fp8_scaled_mm"""
        return self._gemm_a8w8(input_quant, weight, input_scale, weight_scale, bias, dtype)
    
    def int8_scaled_mm(self, input_quant, weight, input_scale, weight_scale, dtype, bias=None):
        """INT8 GEMM compatible with sgl_kernel.int8_scaled_mm"""
        return self._gemm_a8w8(input_quant, weight, input_scale, weight_scale, bias, dtype)
    
    def sgl_per_token_quant_fp8(self, x, out, scale):
        """Per-token FP8 quantization compatible with sgl_kernel.sgl_per_token_quant_fp8"""
        q, s = self._pertoken_quant(x, quant_dtype=self._dtypes.fp8)
        out.copy_(q)
        scale.copy_(s)
    
    def sgl_per_token_group_quant_fp8(self, x, out, scale, group_size=128, eps=1e-10, fp8_min=-448.0, fp8_max=448.0):
        """Per-token per-group FP8 quantization compatible with sgl_kernel.sgl_per_token_group_quant_fp8"""
        m, k = x.shape
        x_view = x.view(m, -1, group_size)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(eps)
        q = (x_view * (fp8_max / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, k)
        s = (x_amax / fp8_max).view(m, -1)
        out.copy_(q)
        scale.copy_(s)


def _get_aiter_sgl_kernel():
    """Get aiter-based sgl_kernel compatibility layer."""
    try:
        import aiter
        return AiterSglKernelCompat(aiter)
    except ImportError:
        logger.error(
            f"\n{'='*60}\n"
            f"ERROR: AMD ROCm detected but aiter is not installed.\n"
            f"aiter is REQUIRED for LightX2V to work on AMD GPUs.\n"
            f"\nPlease install aiter:\n"
            f"{AITER_INSTALL_CMD}\n"
            f"{'='*60}\n"
        )
        raise ImportError(
            "aiter is required for AMD ROCm support. "
            f"Please install: pip install git+{AITER_REPO}@{AITER_COMMIT}"
        )


def init_amd_rocm():
    """
    Initialize AMD ROCm optimizations.
    
    This function should be called early in the import process to:
    1. Disable cudnn for faster VAE convolution
    2. Inject aiter as sgl_kernel compatibility layer (REQUIRED on AMD)
    """
    if not IS_AMD_ROCM:
        return
    
    logger.info("AMD ROCm platform detected, initializing optimizations...")
    
    # Disable cudnn for faster VAE conv computation
    torch.backends.cudnn.enabled = False
    logger.info("  - cudnn disabled for faster VAE convolution")
    
    # Inject aiter as sgl_kernel compatibility layer (REQUIRED)
    sgl_kernel = _get_aiter_sgl_kernel()
    sys.modules["sgl_kernel"] = sgl_kernel
    # Update any module that already imported sgl_kernel
    for mod_name, mod in list(sys.modules.items()):
        if mod is not None and hasattr(mod, 'sgl_kernel'):
            setattr(mod, 'sgl_kernel', sgl_kernel)
    logger.info("  - aiter sgl_kernel compatibility layer enabled (RMSNorm, GEMM)")


@PLATFORM_DEVICE_REGISTER("amd_rocm")
class AmdRocmDevice:
    """
    AMD ROCm Device implementation for LightX2V.

    AMD ROCm uses CUDA-compatible APIs through HIP.
    This class provides AMD-specific optimizations.
    """

    name = "amd_rocm"

    @staticmethod
    def is_available() -> bool:
        """Check if AMD ROCm is available."""
        return IS_AMD_ROCM and torch.cuda.is_available()

    @staticmethod
    def get_device() -> str:
        """Get the device type string. Returns 'cuda' for ROCm compatibility."""
        return "cuda"

    @staticmethod
    def init_parallel_env():
        """Initialize distributed parallel environment for AMD ROCm."""
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())


# Export constants
__all__ = [
    "IS_AMD_ROCM",
    "AITER_REPO", 
    "AITER_COMMIT",
    "AITER_INSTALL_CMD",
    "AiterSglKernelCompat",
    "init_amd_rocm",
    "AmdRocmDevice",
]
