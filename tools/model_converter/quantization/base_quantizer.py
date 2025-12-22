"""
Base quantizer class that all quantizers should inherit from.
"""

import gc
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from tqdm import tqdm


class BaseQuantizer(ABC):
    """
    Abstract base class for weight quantization.
    
    Each quantization method (INT8, FP8, etc.) should implement this interface.
    """

    def __init__(
        self,
        target_modules: Optional[List[str]] = None,
        key_idx: int = 2,
        ignore_keys: Optional[List[str]] = None,
        non_linear_dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize quantizer.
        
        Args:
            target_modules: List of module names to quantize (e.g., ["attn", "ffn"])
            key_idx: Index in key split for module identification
            ignore_keys: Keys to skip during quantization
            non_linear_dtype: Data type for non-quantized layers
        """
        self.target_modules = target_modules or []
        self.key_idx = key_idx
        self.ignore_keys = ignore_keys or []
        self.non_linear_dtype = non_linear_dtype
        
        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def quantize_weight(
        self, weight: torch.Tensor, per_channel: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Quantize a single weight tensor.
        
        Args:
            weight: Weight tensor to quantize (must be 2D)
            per_channel: Use per-channel quantization
            
        Returns:
            Tuple of:
                - Quantized weight tensor
                - Scale tensor
                - Extra info dict (e.g., zero points, global scales)
        """
        pass

    def quantize_model(
        self,
        weights: Dict[str, torch.Tensor],
        adapter_keys: Optional[List[str]] = None,
        comfyui_mode: bool = False,
        comfyui_keys: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize all applicable weights in model.
        
        Args:
            weights: Model state dictionary (modified in-place)
            adapter_keys: Special adapter keys to quantize
            comfyui_mode: ComfyUI compatibility mode
            comfyui_keys: Specific keys to quantize in ComfyUI mode
            
        Returns:
            Modified state dictionary with quantized weights and scales
        """
        total_quantized = 0
        original_size = 0
        quantized_size = 0
        non_quantized_size = 0
        
        keys = list(weights.keys())
        
        with tqdm(keys, desc=f"Quantizing with {self.__class__.__name__}") as pbar:
            for key in pbar:
                pbar.set_postfix(current_key=key[:50], refresh=False)
                
                # Skip ignored keys
                if any(ig_key in key for ig_key in self.ignore_keys):
                    del weights[key]
                    continue
                
                tensor = weights[key]
                
                # Skip non-tensors and non-2D tensors
                if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
                    if tensor.dtype != self.non_linear_dtype:
                        weights[key] = tensor.to(self.non_linear_dtype)
                        non_quantized_size += weights[key].numel() * weights[key].element_size()
                    else:
                        non_quantized_size += tensor.numel() * tensor.element_size()
                    continue
                
                # Check if key matches target modules
                should_quantize = self._should_quantize(
                    key, adapter_keys, comfyui_mode, comfyui_keys
                )
                
                if not should_quantize:
                    if tensor.dtype != self.non_linear_dtype:
                        weights[key] = tensor.to(self.non_linear_dtype)
                        non_quantized_size += weights[key].numel() * weights[key].element_size()
                    else:
                        non_quantized_size += tensor.numel() * tensor.element_size()
                    continue
                
                # Quantize tensor
                try:
                    original_tensor_size = tensor.numel() * tensor.element_size()
                    original_size += original_tensor_size
                    
                    w_q, scales, extra = self.quantize_weight(tensor, per_channel=True)
                    
                    # Replace original tensor and store scales
                    weights[key] = w_q
                    if comfyui_mode:
                        weights[key.replace(".weight", ".scale_weight")] = scales
                    else:
                        weights[key + "_scale"] = scales
                    
                    # Store any extra parameters (e.g., global scales for NVFP4)
                    for extra_key, extra_value in extra.items():
                        if extra_value is not None:
                            weights[key + f"_{extra_key}"] = extra_value
                    
                    quantized_tensor_size = w_q.numel() * w_q.element_size()
                    scale_size = scales.numel() * scales.element_size()
                    quantized_size += quantized_tensor_size + scale_size
                    
                    total_quantized += 1
                    del w_q, scales
                    
                except Exception as e:
                    logger.error(f"Error quantizing {key}: {str(e)}")
                    # Keep original weight
                    non_quantized_size += tensor.numel() * tensor.element_size()
                
                gc.collect()
        
        # Log statistics
        self._log_statistics(
            total_quantized,
            original_size,
            quantized_size,
            non_quantized_size,
        )
        
        # Add special markers for ComfyUI if needed
        if comfyui_mode:
            weights = self._add_comfyui_markers(weights)
        
        return weights

    def _should_quantize(
        self,
        key: str,
        adapter_keys: Optional[List[str]],
        comfyui_mode: bool,
        comfyui_keys: Optional[List[str]],
    ) -> bool:
        """Check if a key should be quantized."""
        parts = key.split(".")
        
        # ComfyUI mode special handling
        if comfyui_mode and comfyui_keys and key in comfyui_keys:
            return True
        
        # Check target modules
        if len(parts) >= self.key_idx + 1 and parts[self.key_idx] in self.target_modules:
            return True
        
        # Check adapter keys
        if adapter_keys and any(adapter_key in parts for adapter_key in adapter_keys):
            return True
        
        return False

    def _log_statistics(
        self,
        total_quantized: int,
        original_size: int,
        quantized_size: int,
        non_quantized_size: int,
    ):
        """Log quantization statistics."""
        original_size_mb = original_size / (1024**2)
        quantized_size_mb = quantized_size / (1024**2)
        non_quantized_size_mb = non_quantized_size / (1024**2)
        total_final_size_mb = (quantized_size + non_quantized_size) / (1024**2)
        
        if original_size > 0:
            size_reduction_mb = original_size_mb - quantized_size_mb
            reduction_pct = (size_reduction_mb / original_size_mb) * 100
        else:
            size_reduction_mb = 0
            reduction_pct = 0
        
        logger.info(f"Quantized {total_quantized} tensors")
        logger.info(f"Original quantized tensors size: {original_size_mb:.2f} MB")
        logger.info(f"After quantization size: {quantized_size_mb:.2f} MB (includes scales)")
        logger.info(f"Non-quantized tensors size: {non_quantized_size_mb:.2f} MB")
        logger.info(f"Total final model size: {total_final_size_mb:.2f} MB")
        logger.info(f"Size reduction: {size_reduction_mb:.2f} MB ({reduction_pct:.1f}%)")

    def _add_comfyui_markers(
        self, weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add special markers for ComfyUI compatibility."""
        # Default: no markers needed
        return weights


    def validate_weight(self, weight: torch.Tensor):
        """
        Validate weight tensor before quantization.
        
        Args:
            weight: Weight tensor to validate
            
        Raises:
            ValueError: If weight is invalid
        """
        if weight.dim() != 2:
            raise ValueError(f"Only 2D tensors supported. Got {weight.dim()}D tensor")
        
        if torch.isnan(weight).any():
            raise ValueError("Tensor contains NaN values")
        
        if torch.isinf(weight).any():
            raise ValueError("Tensor contains Inf values")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"target_modules={self.target_modules})"
        )

