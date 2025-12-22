"""MxFP (Microscaling Floating Point) quantization implementations."""

from typing import Any, Dict, Tuple

import torch

from ..core.registry import QUANTIZER_REGISTRY
from .base_quantizer import BaseQuantizer

try:
    from lightx2v_kernel.gemm import (
        scaled_mxfp4_quant,
        scaled_mxfp6_quant,
        scaled_mxfp8_quant,
    )
    MXFP_AVAILABLE = True
except ImportError:
    MXFP_AVAILABLE = False


class MxFPQuantizer(BaseQuantizer):
    """Base class for MxFP quantizers."""

    def __init__(self, *args, **kwargs):
        """Initialize MxFP quantizer."""
        super().__init__(*args, **kwargs)
        
        if not MXFP_AVAILABLE:
            raise ImportError(
                "lightx2v_kernel is required for MxFP quantization. "
                "Please install it from the project."
            )

    def _quantize_mxfp(
        self, 
        weight: torch.Tensor, 
        quant_func
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Common MxFP quantization logic.
        
        Args:
            weight: Weight tensor
            quant_func: Quantization function (mxfp4/6/8)
            
        Returns:
            Tuple of (quantized_weight, scales, extra_info)
        """
        self.validate_weight(weight)
        
        device = weight.device
        
        # MxFP requires bfloat16 input and CUDA
        weight = weight.cuda().to(torch.bfloat16)
        
        # Quantize
        w_q, scales = quant_func(weight)
        
        # Move back to original device
        w_q = w_q.to(device)
        scales = scales.to(device)
        
        return w_q, scales, {}


@QUANTIZER_REGISTRY.register("mxfp4")
class MxFP4Quantizer(MxFPQuantizer):
    """MxFP4 quantization."""

    def quantize_weight(
        self, weight: torch.Tensor, per_channel: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Quantize weight to MxFP4."""
        return self._quantize_mxfp(weight, scaled_mxfp4_quant)


@QUANTIZER_REGISTRY.register("mxfp6")
class MxFP6Quantizer(MxFPQuantizer):
    """MxFP6 quantization."""

    def quantize_weight(
        self, weight: torch.Tensor, per_channel: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Quantize weight to MxFP6."""
        return self._quantize_mxfp(weight, scaled_mxfp6_quant)


@QUANTIZER_REGISTRY.register("mxfp8")
class MxFP8Quantizer(MxFPQuantizer):
    """MxFP8 quantization."""

    def quantize_weight(
        self, weight: torch.Tensor, per_channel: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Quantize weight to MxFP8."""
        return self._quantize_mxfp(weight, scaled_mxfp8_quant)

