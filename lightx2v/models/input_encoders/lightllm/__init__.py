"""
LightLLM-optimized Text Encoder implementation
Extracts core inference optimizations from LightLLM for LightX2V integration

Available Encoders:
1. LightLLMServiceTextEncoder - 通过 HTTP 服务调用 LightLLM（需要独立服务）
2. LightLLMKernelTextEncoder - 基于 HuggingFace 模型 + Triton Kernels 优化
"""

from .qwen25_text_encoder_kernel import LightLLMKernelTextEncoder
from .qwen25_text_encoder_service import LightLLMServiceTextEncoder

__all__ = [
    "LightLLMServiceTextEncoder",  # Service模式：通过HTTP调用LightLLM服务
    "LightLLMKernelTextEncoder",  # Kernel模式：HF模型 + Triton kernels
]
