"""
AMD ROCm Device implementation for LightX2V.

AMD ROCm provides CUDA-compatible APIs through HIP (Heterogeneous-computing Interface for Portability).
This module handles AMD-specific optimizations including:
- Disabling cudnn for faster VAE convolution
- sgl_kernel compatibility layer using aiter library (optional; only for the aiter-backed GEMM/RMSNorm path)
- Optional run-to-run deterministic FP32 GEMM/convolution (ROCM_DETERMINISTIC_FP32_BLAS=1)
"""

import functools
import os
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
        # aiter is optional: it only backs the aiter_attn / sgl_kernel GEMM +
        # RMSNorm path. Torch-native paths (e.g. torch._scaled_mm / torch._int_mm
        # fp8/int8 GEMM, torch_sdpa / SageAttention) run without it. Warn and let
        # the caller skip the sgl_kernel injection rather than aborting setup.
        logger.warning(f"aiter not installed; skipping the aiter sgl_kernel/RMSNorm compatibility layer. This is fine for torch-native paths. To enable the aiter-backed path:\n{AITER_INSTALL_CMD}")
        return None


def _fp32_on_rocblas(fn):
    """Run ``fn`` on rocBLAS when one of its first two arguments is an fp32 GPU tensor.

    hipBLASLt may pick a different FP32 GEMM kernel in each process for the same
    shape (e.g. MT16x8x64 on one rank, MT128x128x16 on another), so FP32 layers give
    ulp-level different results from run to run and diffusion sampling amplifies them.
    rocBLAS kernel selection is fixed per shape. BF16/FP16 GEMMs keep using hipBLASLt.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not any(isinstance(t, torch.Tensor) and t.is_cuda and t.dtype == torch.float32 for t in args[:2]):
            return fn(*args, **kwargs)
        prev = torch.backends.cuda.preferred_blas_library()
        torch.backends.cuda.preferred_blas_library("cublas")  # rocBLAS on ROCm
        try:
            return fn(*args, **kwargs)
        finally:
            torch.backends.cuda.preferred_blas_library(prev)

    return wrapper


def _route_fp32_blas_to_rocblas():
    # rocBLAS itself may hand GEMMs to hipBLASLt on recent ROCm; keep them in rocBLAS.
    os.environ.setdefault("ROCBLAS_USE_HIPBLASLT", "0")
    F = torch.nn.functional
    for mod, name in ((torch, "mm"), (torch, "addmm"), (F, "linear"), (F, "conv1d"), (F, "conv2d"), (F, "conv3d")):
        fn = getattr(mod, name)
        if not getattr(fn, "_lightx2v_fp32_rocblas", False):
            wrapped = _fp32_on_rocblas(fn)
            wrapped._lightx2v_fp32_rocblas = True
            setattr(mod, name, wrapped)


@PLATFORM_DEVICE_REGISTER("amd_rocm")
class AmdRocmDevice:
    """
    AMD ROCm Device implementation for LightX2V.

    AMD ROCm uses CUDA-compatible APIs through HIP.
    This class provides AMD-specific optimizations.
    """

    name = "amd_rocm"

    @staticmethod
    def init_device_env():
        """
        Initialize AMD ROCm optimizations.

        This is called from lightx2v_platform.set_ai_device when platform is amd_rocm.
        1. Disable cudnn for faster VAE convolution
        2. Inject aiter as sgl_kernel compatibility layer (optional; only the
           aiter-backed GEMM/RMSNorm path needs it)
        3. With ROCM_DETERMINISTIC_FP32_BLAS=1, run FP32 GEMMs/convolutions on rocBLAS
           so that repeated runs give bitwise identical outputs
        """
        logger.info("AMD ROCm platform detected, initializing optimizations...")

        # Disable cudnn for faster VAE conv computation
        torch.backends.cudnn.enabled = False
        logger.info("  - cudnn disabled for faster VAE convolution")

        # Inject aiter as sgl_kernel compatibility layer when available. Optional:
        # torch-native paths (torch._scaled_mm / torch._int_mm fp8/int8 GEMM,
        # torch_sdpa / SageAttention) run without aiter.
        sgl_kernel = _get_aiter_sgl_kernel()
        if sgl_kernel is not None:
            sys.modules["sgl_kernel"] = sgl_kernel
            # Update any module that already imported sgl_kernel. torch.ops and
            # torch.classes are ModuleType subclasses living in sys.modules whose
            # attributes are operator namespaces rather than imported modules, so
            # overwriting them would replace torch.ops.sgl_kernel itself.
            for mod_name, mod in list(sys.modules.items()):
                if mod_name in ("torch.ops", "torch.classes"):
                    continue
                if mod is not None and hasattr(mod, "sgl_kernel"):
                    setattr(mod, "sgl_kernel", sgl_kernel)
            logger.info("  - aiter sgl_kernel compatibility layer enabled (RMSNorm, GEMM)")

        if os.getenv("ROCM_DETERMINISTIC_FP32_BLAS", "0") in ("1", "True", "true"):
            _route_fp32_blas_to_rocblas()
            logger.info("  - FP32 GEMM/convolution routed to rocBLAS (deterministic across runs)")

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
    "AmdRocmDevice",
]
