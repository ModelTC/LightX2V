"""
Intel XPU quantized linear layers for text encoders (T5, CLIP, etc.)

These are nn.Module-based quantized linear layers optimized for Intel XPU.
"""

import torch
import torch.nn as nn
from loguru import logger


class IntelXpuQuantLinearFp8(nn.Module):
    """
    Intel XPU FP8 quantized linear layer for text encoders.

    Strategy:
        - Storage: FP8 (torch.float8_e4m3fn) - saves 50% memory
        - Computation: FP16 using PyTorch native ops
        - Dynamically dequantize FP8 → FP16 during forward pass

    Usage:
        Used in T5 text encoder when config has:
        { 
            "t5_cpu_offload": false,
            "t5_quantized": true,
            "t5_quant_scheme": "fp8-intel-xpu"
        }
    """

    def __init__(self, in_features, out_features, bias=True, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        # Register FP8 weight buffer
        self.register_buffer(
            "weight",
            torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn)
        )

        # Register FP32 scale buffer (per-channel)
        self.register_buffer(
            "weight_scale",
            torch.empty((out_features, 1), dtype=torch.float32)
        )

        # Register bias buffer
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=dtype))
        else:
            self.register_buffer("bias", None)

    def forward(self, input_tensor):
        """
        Forward pass with FP8 → FP16 dequantization

        Args:
            input_tensor: Input tensor, shape [batch, seq_len, in_features] or [1, batch, in_features]

        Returns:
            Output tensor, shape [batch, seq_len, out_features] or [1, batch, out_features]
        """
       
        # Handle T5-style input: [1, batch, features] → [batch, features]
        squeeze_output = False
        if input_tensor.dim() == 3 and input_tensor.shape[0] == 1:
            input_tensor = input_tensor.squeeze(0)
            squeeze_output = True

        # Ensure input is FP16
        if input_tensor.dtype != self.dtype:
            input_tensor = input_tensor.to(self.dtype)

        # Dequantize weight: FP8 → FP16
        # weight shape: [out_features, in_features] in FP8
        # weight_scale shape: [out_features, 1] for per-channel scale
        weight_fp16 = self.weight.to(self.dtype) * self.weight_scale.to(self.dtype)

        # Handle bias
        bias_fp16 = self.bias.to(self.dtype) if self.bias is not None else None

        # Compute using PyTorch native linear (Intel XPU compatible)
        output = torch.nn.functional.linear(input_tensor, weight_fp16, bias_fp16)

        # Restore original shape if needed
        if squeeze_output:
            output = output.unsqueeze(0)

        return output

    def _apply(self, fn):
        """
        Override _apply to handle device movement (CPU ↔ XPU)

        This is called when .to(device) or .cuda() is invoked.
        """
        for module in self.children():
            module._apply(fn)

        def maybe_cast(t):
            if t is not None and t.device != fn(t).device:
                return fn(t)
            return t

        self.weight = maybe_cast(self.weight)
        self.weight_scale = maybe_cast(self.weight_scale)
        self.bias = maybe_cast(self.bias)

        return self

    def __repr__(self):
        return (
            f"IntelXpuQuantLinearFp8("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"dtype={self.dtype})"
        )


class IntelXpuQuantLinearInt8(nn.Module):
    """
    Intel XPU INT8 quantized linear layer for text encoders.

    Strategy:
        - Storage: INT8 - saves 50% memory
        - Computation: FP16 using PyTorch native ops
        - Dynamically dequantize INT8 → FP16 during forward pass

    Usage:
        Used in T5 text encoder when config has:
        {
            "t5_quantized": true,
            "t5_quant_scheme": "int8-intel-xpu"
        }
    """

    def __init__(self, in_features, out_features, bias=True, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        # Register INT8 weight buffer
        self.register_buffer(
            "weight",
            torch.empty((out_features, in_features), dtype=torch.int8)
        )

        # Register FP32 scale buffer (per-channel)
        self.register_buffer(
            "weight_scale",
            torch.empty((out_features, 1), dtype=torch.float32)
        )

        # Register bias buffer
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=dtype))
        else:
            self.register_buffer("bias", None)

    def forward(self, input_tensor):
        """
        Forward pass with INT8 → FP16 dequantization
        """
        # Handle T5-style input
        squeeze_output = False
        if input_tensor.dim() == 3 and input_tensor.shape[0] == 1:
            input_tensor = input_tensor.squeeze(0)
            squeeze_output = True

        # Ensure input is FP16
        if input_tensor.dtype != self.dtype:
            input_tensor = input_tensor.to(self.dtype)

        # Dequantize weight: INT8 → FP16
        weight_fp16 = self.weight.to(self.dtype) * self.weight_scale.to(self.dtype)

        # Handle bias
        bias_fp16 = self.bias.to(self.dtype) if self.bias is not None else None

        # Compute
        output = torch.nn.functional.linear(input_tensor, weight_fp16, bias_fp16)

        # Restore shape
        if squeeze_output:
            output = output.unsqueeze(0)

        return output

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        def maybe_cast(t):
            if t is not None and t.device != fn(t).device:
                return fn(t)
            return t

        self.weight = maybe_cast(self.weight)
        self.weight_scale = maybe_cast(self.weight_scale)
        self.bias = maybe_cast(self.bias)

        return self

    def __repr__(self):
        return (
            f"IntelXpuQuantLinearInt8("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"dtype={self.dtype})"
        )
