"""
Intel XPU matrix multiplication using torch.xpu.
"""

import re
import os
from abc import ABCMeta, abstractmethod

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.ops.mm.template import MMWeightTemplate, MMWeightQuantTemplate
from lightx2v_platform.registry_factory import PLATFORM_MM_WEIGHT_REGISTER

# Detect Intel XPU platform
IS_INTEL_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()

DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "FP16": torch.float16,
    "FP32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
}


def GET_DTYPE():
    RUNNING_FLAG = os.getenv("DTYPE", "BF16")
    assert RUNNING_FLAG in ["BF16", "FP16"]
    return DTYPE_MAP[RUNNING_FLAG]


@PLATFORM_MM_WEIGHT_REGISTER("intel_xpu_mm")
class IntelXpuMmWeight(MMWeightTemplate):
    """
    Intel XPU matrix multiplication implementation.
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, is_post_adapter)

    def load(self, weight_dict):
        if self.create_cuda_buffer:
            self._load_cuda_buffers(weight_dict)
        elif self.create_cpu_buffer:
            self._load_cpu_pin_buffers()
        else:
            self._load_default_tensors(weight_dict)

    def _load_default_tensors(self, weight_dict):
        if not self.lazy_load:
            device = weight_dict[self.weight_name].device
            if device.type == "cpu":
                weight_tensor = weight_dict[self.weight_name]
                self.pin_weight = self._create_cpu_pin_weight(weight_tensor)
                if self.bias_name is not None and self.bias_name in weight_dict:
                    bias_tensor = weight_dict[self.bias_name]
                    self.pin_bias = self._create_cpu_pin_weight(bias_tensor)
                else:
                    self.pin_bias = None
                del weight_dict[self.weight_name]
            else:
                self.weight = weight_dict[self.weight_name]
                self.bias = weight_dict[self.bias_name] if self.bias_name is not None and self.bias_name in weight_dict else None
        else:
            self.weight = None
            self.bias = None

    def _get_weight_tensor(self, weight_dict=None):
        if self.lazy_load:
            with safe_open(self.lazy_load_file, framework="pt", device="cpu") as lazy_load_file:
                tensor = lazy_load_file.get_tensor(self.weight_name)
        else:
            tensor = weight_dict[self.weight_name]
        return tensor

    def _create_cpu_pin_weight(self, tensor):
        if tensor is None:
            return None
        pin_tensor = torch.empty(tensor.shape, pin_memory=True, dtype=tensor.dtype)
        pin_tensor.copy_(tensor)
        del tensor
        return pin_tensor

    def _load_cuda_buffer(self, weight_dict):
        weight_tensor = self._get_weight_tensor(weight_dict)
        self.weight_cuda_buffer = weight_tensor.to(AI_DEVICE)

        if self.bias_name is not None and self.bias_name in weight_dict:
            self.bias_cuda_buffer = weight_dict[self.bias_name].to(AI_DEVICE)

    def _load_cpu_pin_buffer(self):
        weight_tensor = self._get_weight_tensor()
        self.pin_weight = self._create_cpu_pin_weight(weight_tensor)

    def apply(self, input_tensor):
        if hasattr(self, "weight_cuda_buffer"):
            weight = self.weight_cuda_buffer
            bias = self.bias_cuda_buffer if hasattr(self, "bias_cuda_buffer") else None
        elif hasattr(self, "weight"):
            weight = self.weight
            bias = self.bias
        else:
            weight = self.pin_weight.to(AI_DEVICE)
            bias = self.pin_bias.to(AI_DEVICE) if self.pin_bias is not None else None

        output = torch.nn.functional.linear(input_tensor, weight, bias)
        return output


@PLATFORM_MM_WEIGHT_REGISTER("intel_xpu_fp8")
class IntelXpuFp8MmWeight(MMWeightQuantTemplate):
    """
    Intel XPU FP8 quantized matrix multiplication.

    Strategy:
        - Storage: FP8 (torch.float8_e4m3fn) - saves 50% memory
        - Computation: FP16 using PyTorch native ops
        - Dynamically dequantize FP8 → FP16 during forward pass

    Benefits:
        - Memory efficient: FP8 storage (8-bit)
        - Compatible: FP16 compute using PyTorch native ops
        - Intel XPU friendly: No CUDA-specific kernels
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.weight_scale_name = self.weight_name.removesuffix(".weight") + ".weight_scale"
        self.load_func = self.load_fp8_perchannel_sym
        self.weight_need_transpose = False  # Handle transpose in apply
        self.infer_dtype = torch.float16

    def load_fp8_perchannel_sym(self, weight_dict):
        """Load FP8 quantized weights with per-channel symmetric quantization"""
        if self.config.get("weight_auto_quant", False):
            # Auto quantize from FP16/FP32 to FP8
            self.weight = weight_dict[self.weight_name].to(torch.float32)

            # Per-channel quantization
            # weight shape: [in_features, out_features] or [out_features, in_features]
            # Calculate scale per output channel
            weight_abs_max = self.weight.abs().max(dim=0, keepdim=True)[0]
            self.weight_scale = weight_abs_max / 448.0  # FP8 E4M3 max value ≈ 448
            self.weight_scale = self.weight_scale.clamp(min=1e-12)  # Avoid division by zero

            # Quantize to FP8
            weight_normalized = self.weight / self.weight_scale
            self.weight = weight_normalized.to(torch.float8_e4m3fn)
            self.weight_scale = self.weight_scale.squeeze(0).to(torch.float32)
        else:
            # Load pre-quantized weights
            self.load_quantized(weight_dict)

    def load(self, weight_dict):
        """Load weights and transpose if needed"""
        self.load_quantized(weight_dict)
        # Note: No transpose here, will handle in apply()

    def apply(self, input_tensor):
        """
        Forward pass with FP8 → FP16 dequantization

        Steps:
        1. Ensure input is FP16
        2. Dequantize weight: fp8 → fp16 (weight * scale)
        3. Compute: torch.nn.functional.linear(input_fp16, weight_fp16, bias)
        """
        # Ensure input is FP16
        if input_tensor.dtype != torch.float16:
            input_tensor = input_tensor.to(torch.float16)

        # Dequantize weight: FP8 → FP16
        # weight shape: [in_features, out_features] before transpose
        # weight_scale shape: [out_features] for per-channel quantization
        weight_fp16 = self.weight.to(torch.float16)

        # Apply per-channel scale
        if self.weight_scale.dim() == 1:
            # weight: [in_features, out_features]
            # scale each output channel (columns)
            weight_fp16 = weight_fp16 * self.weight_scale.unsqueeze(0)
        else:
            # weight_scale: [out_features, 1] or [1, out_features]
            weight_fp16 = weight_fp16 * self.weight_scale.t()

        # Transpose weight for F.linear (expects [out_features, in_features])
        weight_fp16 = weight_fp16.t()

        # Handle bias
        bias_fp16 = None
        if self.bias is not None:
            bias_fp16 = self.bias.to(torch.float16)

        # Use PyTorch native linear (Intel XPU compatible)
        output = torch.nn.functional.linear(input_tensor, weight_fp16, bias_fp16)

        return output
