"""Wan block linear conversion for the SM120 FP16 accumulator contract."""

import torch

from lightx2v.models.networks.wan.fp8_f16_accum_policy import QUANTIZATION_PROFILE, WEIGHT_QMAX


class WanFP8F16AccumQuantization:
    def __init__(self, model_type):
        if model_type not in {"wan_dit", "self_forcing"}:
            raise ValueError(f"{QUANTIZATION_PROFILE} requires model_type=wan_dit or self_forcing")
        self.count = 0

    @property
    def metadata(self):
        return {"format": "pt", "quantization_profile": QUANTIZATION_PROFILE, "weight_qmax": str(WEIGHT_QMAX)}

    def quantize_weight(self, name, weight, default_quantize):
        if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError(f"{name} must be an original floating-point weight, got {weight.dtype}")
        values = weight.to(torch.float32, copy=True)
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} contains nonfinite values")
        scales = values.abs().amax(dim=1, keepdim=True).clamp_min_(1e-8).div_(WEIGHT_QMAX)
        quantized = values.div_(scales).clamp_(-WEIGHT_QMAX, WEIGHT_QMAX).to(torch.float8_e4m3fn)
        self.count += 1
        return quantized, scales, {}

    def validate(self):
        if self.count == 0:
            raise ValueError("No Wan block linears were selected for FP8-F16 accumulation conversion")
