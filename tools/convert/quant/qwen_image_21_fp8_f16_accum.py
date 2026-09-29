"""Qwen-Image-2.1 block linear conversion for FP16 accumulation."""

import torch

from lightx2v.models.networks.qwen_image_21.fp8_f16_accum_policy import QUANTIZATION_PROFILE, WEIGHT_QMAX


class QwenImage21FP8F16AccumQuantization:
    def __init__(self, model_type):
        if model_type != "qwen_image_21_dit":
            raise ValueError(f"{QUANTIZATION_PROFILE} requires model_type=qwen_image_21_dit")
        self.count = 0

    @property
    def metadata(self):
        return {"format": "pt", "quantization_profile": QUANTIZATION_PROFILE, "weight_qmax": str(WEIGHT_QMAX)}

    def quantize_weight(self, name, weight, default_quantize):
        values = weight.float()
        scales = values.abs().amax(dim=1, keepdim=True).clamp_min_(1e-8).div_(WEIGHT_QMAX)
        quantized = values.div(scales).clamp_(-WEIGHT_QMAX, WEIGHT_QMAX).to(torch.float8_e4m3fn)
        self.count += 1
        return quantized, scales, {}

    def validate(self):
        if self.count != 224:
            raise ValueError(f"Expected 224 Qwen-Image-2.1 block linears, got {self.count}")
