"""
Iluvatar GPU quantized linear layers for text encoders (T5, CLIP, etc.)

These are nn.Module-based quantized linear layers optimized for Iluvatar GPU
"""

import torch
import torch.nn as nn

try:
    import ixformer.inference.functions as ixf
except ImportError:
    ixf = None


class IluvatarQuantLinearInt8(nn.Module):
    """
    Iluvatar GPU INT8 quantized linear layer for text encoders.

    Strategy:
        - Storage: INT8 - saves 50% memory
        - Computation: FP16 using PyTorch native ops
        - Dynamically dequantize INT8 → FP16 during forward pass

    Usage:
        Used in T5 text encoder when config has:
        {
            "t5_quantized": true,
            "t5_quant_scheme": "int8-iluvatar-cuda"
        }
    """

    def __init__(self, in_features, out_features, bias=True, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        assert ixf is not None, "iluvatar ixformer is not installed."
        # Register INT8 weight buffer
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))

        # Register FP32 scale buffer (per-channel)
        self.register_buffer("weight_scale", torch.empty((out_features, 1), dtype=torch.float32))

        # Register bias buffer
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=dtype))
        else:
            self.register_buffer("bias", None)

    def act_quant_func(self, x):
        input_tensor_quant, input_tensor_scale = ixf.dynamic_scaled_int8_quant(x)
        return input_tensor_quant, input_tensor_scale

    def forward(self, input_tensor):
        """
        Forward pass with INT8 → FP16 dequantization
        """
        # Handle T5-style input
        squeeze_output = False
        dtype = input_tensor.dtype
        if input_tensor.dim() == 3 and input_tensor.shape[0] == 1:
            input_tensor = input_tensor.squeeze(0)
            squeeze_output = True

        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output = ixf.w8a8(input=input_tensor_quant, weight=self.weight, i_scales=input_tensor_scale, w_scales=self.weight_scale.reshape(-1), bias=self.bias, out_dtype=dtype)

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
        return f"IluvatarQuantLinearInt8(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, dtype={self.dtype})"
