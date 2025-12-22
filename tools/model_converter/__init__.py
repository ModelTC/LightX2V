"""
LightX2V Model Converter - 新一代模型转换工具

统一的模型转换框架，支持：
- 多种量化格式（INT8, FP8, NVFP4, MxFP等）
- LoRA融合
- 格式转换（Diffusers ↔ LightX2V）
- 多硬件后端（NVIDIA, DCU, Ascend等）
"""

__version__ = "2.0.0"

from .core.registry import (
    CONVERTER_REGISTRY,
    FORMAT_REGISTRY,
    QUANTIZER_REGISTRY,
)

__all__ = [
    "CONVERTER_REGISTRY",
    "FORMAT_REGISTRY",
    "QUANTIZER_REGISTRY",
]

