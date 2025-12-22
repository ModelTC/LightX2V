"""NVIDIA FP4 quantization implementation."""

from typing import Any, Dict, Tuple

import torch

from ..core.registry import QUANTIZER_REGISTRY
from .base_quantizer import BaseQuantizer

try:
    from lightx2v_kernel.gemm import scaled_nvfp4_quant
    NVFP4_AVAILABLE = True
except ImportError:
    NVFP4_AVAILABLE = False


@QUANTIZER_REGISTRY.register("nvfp4")
class NVFP4Quantizer(BaseQuantizer):
    """NVIDIA FP4 quantization."""

    def __init__(self, *args, **kwargs):
        """Initialize NVFP4 quantizer."""
        super().__init__(*args, **kwargs)
        
        if not NVFP4_AVAILABLE:
            raise ImportError(
                "lightx2v_kernel is required for NVFP4 quantization. "
                "Please install it from the project."
            )

    def quantize_weight(
        self, weight: torch.Tensor, per_channel: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Quantize weight to NVIDIA FP4.
        
        Args:
            weight: Weight tensor (2D)
            per_channel: Use per-channel quantization
            
        Returns:
            Tuple of (quantized_weight, scales, extra_info with global_scale)
        """
        self.validate_weight(weight)
        
        device = weight.device
        
        # NVFP4 requires bfloat16 input and CUDA
        weight = weight.cuda().to(torch.bfloat16)
        
        # Calculate global scale
        weight_global_scale = (2688.0 / torch.max(torch.abs(weight))).to(torch.float32)
        
        # Quantize
        w_q, scales = scaled_nvfp4_quant(weight, weight_global_scale)
        
        # Move back to original device
        w_q = w_q.to(device)
        scales = scales.to(device)
        weight_global_scale = weight_global_scale.to(device)
        
        # Return global scale in extra info
        extra = {"weight_global_scale": weight_global_scale}
        
        return w_q, scales, extra

