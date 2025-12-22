"""FP8 quantization implementation."""

from typing import Any, Dict, Tuple

import torch

from ..core.registry import QUANTIZER_REGISTRY
from .base_quantizer import BaseQuantizer

try:
    from qtorch.quant import float_quantize
except ImportError:
    float_quantize = None


@QUANTIZER_REGISTRY.register("fp8")
class FP8Quantizer(BaseQuantizer):
    """FP8 E4M3 quantization."""

    def __init__(self, *args, **kwargs):
        """Initialize FP8 quantizer."""
        super().__init__(*args, **kwargs)
        
        if float_quantize is None:
            raise ImportError(
                "qtorch is required for FP8 quantization. "
                "Please install it with: pip install qtorch"
            )

    def quantize_weight(
        self, weight: torch.Tensor, per_channel: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Quantize weight to FP8 E4M3.
        
        Args:
            weight: Weight tensor (2D)
            per_channel: Use per-channel quantization
            
        Returns:
            Tuple of (quantized_weight, scales, extra_info)
        """
        self.validate_weight(weight)
        
        org_w_shape = weight.shape
        
        # Calculate scale
        if per_channel:
            max_val = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        else:
            max_val = weight.abs().max()
        
        finfo = torch.finfo(torch.float8_e4m3fn)
        qmin, qmax = finfo.min, finfo.max
        scales = max_val / qmax
        
        # Scale and clip
        scaled_tensor = weight / scales
        scaled_tensor = torch.clip(scaled_tensor, qmin, qmax)
        
        # Quantize using qtorch
        w_q = float_quantize(scaled_tensor.float(), 4, 3, rounding="nearest").to(torch.float8_e4m3fn)
        
        # Validate
        assert torch.isnan(scales).sum() == 0, "NaN in scales"
        assert torch.isnan(w_q).sum() == 0, "NaN in quantized weights"
        
        if per_channel:
            scales = scales.view(org_w_shape[0], -1)
            w_q = w_q.reshape(org_w_shape)
        
        return w_q, scales, {}

    def _add_comfyui_markers(
        self, weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add FP8 marker for ComfyUI."""
        # ComfyUI expects a special marker tensor
        weights["scaled_fp8"] = torch.zeros(2, dtype=torch.float8_e4m3fn)
        return weights

