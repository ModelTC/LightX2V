"""INT8 quantization implementation."""

from typing import Any, Dict, Tuple

import torch

from ..core.registry import QUANTIZER_REGISTRY
from .base_quantizer import BaseQuantizer


@QUANTIZER_REGISTRY.register("int8")
class INT8Quantizer(BaseQuantizer):
    """INT8 symmetric per-channel quantization."""

    def quantize_weight(
        self, weight: torch.Tensor, per_channel: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Quantize weight to INT8.
        
        Uses symmetric quantization: [-128, 127]
        
        Args:
            weight: Weight tensor (2D)
            per_channel: Use per-channel quantization (per output channel)
            
        Returns:
            Tuple of (quantized_weight, scales, extra_info)
        """
        self.validate_weight(weight)
        
        org_w_shape = weight.shape
        
        # Calculate scale per output channel
        if per_channel:
            max_val = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        else:
            max_val = weight.abs().max()
        
        qmin, qmax = -128, 127
        scales = max_val / qmax
        
        # Quantize
        w_q = torch.clamp(torch.round(weight / scales), qmin, qmax).to(torch.int8)
        
        # Validate
        assert torch.isnan(scales).sum() == 0, "NaN in scales"
        assert torch.isnan(w_q).sum() == 0, "NaN in quantized weights"
        
        if per_channel:
            scales = scales.view(org_w_shape[0], -1)
            w_q = w_q.reshape(org_w_shape)
        
        return w_q, scales, {}

