"""
AMD ROCm optimized attention using aiter library.
Provides significantly faster attention computation on AMD GPUs (2.5x-6x speedup).
Internally uses FA3 (fmha_v3) when conditions are met.
"""

import torch
from loguru import logger

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

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

# Try to import aiter (AMD ROCm optimized)
aiter_flash_attn_varlen_func = None
AITER_AVAILABLE = False
AITER_IMPORT_ERROR = None

try:
    from aiter import flash_attn_varlen_func as aiter_flash_attn_varlen_func

    AITER_AVAILABLE = True
    logger.info("aiter flash_attn_varlen_func found (AMD ROCm optimized)")
except ImportError as e:
    AITER_IMPORT_ERROR = str(e)
    if IS_AMD_ROCM:
        logger.warning(
            f"aiter not found on AMD ROCm platform. "
            f"For optimal performance, please install aiter:\n{AITER_INSTALL_CMD}"
        )
    else:
        logger.debug("aiter not found (only available on AMD ROCm platform)")


@PLATFORM_ATTN_WEIGHT_REGISTER("aiter_attn")
class AiterAttnWeight(AttnWeightTemplate):
    """
    AMD ROCm optimized attention using aiter library.

    Performance:
        - 2.5x-6x faster than flash_attn package on AMD GPUs
        - Automatically uses FA3 (fmha_v3) when conditions are met

    Requirements:
        - aiter library (AMD ROCm)
        - AMD GPU with ROCm support
    """

    def __init__(self):
        self.config = {}
        
        # Check platform first
        if not IS_AMD_ROCM:
            raise RuntimeError(
                "aiter_attn is only available on AMD ROCm platform.\n"
                "Current platform is not AMD ROCm (torch.version.hip is not set).\n"
                "For NVIDIA GPUs, please use 'flash_attn2' or 'flash_attn3' instead."
            )
        
        # Check aiter availability
        if not AITER_AVAILABLE:
            raise ImportError(
                f"aiter is not installed on AMD ROCm platform.\n"
                f"Import error: {AITER_IMPORT_ERROR}\n"
                f"Please install aiter for optimal performance:\n{AITER_INSTALL_CMD}"
            )

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        model_cls=None,
    ):
        if len(q.shape) == 3:
            bs = 1
        elif len(q.shape) == 4:
            bs = q.shape[0]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])

        x = aiter_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ).reshape(bs * max_seqlen_q, -1)
        return x

