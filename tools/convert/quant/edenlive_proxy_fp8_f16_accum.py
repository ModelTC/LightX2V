"""Quantize ProxyDINO linears for the reduced-range FP16 accumulator."""

import torch

from lightx2v.utils.fp8_f16_accum_checkpoint import ACTIVATION_QMAX, WEIGHT_QMAX
from lightx2v.utils.fp8_f16_accum_checkpoint import EDENLIVE_PROXY_QUANTIZATION_PROFILE as QUANTIZATION_PROFILE


class EdenLiveProxyFP8F16AccumQuantization:
    def __init__(self):
        self.count = 0

    @property
    def metadata(self):
        return {"quantization_profile": QUANTIZATION_PROFILE, "weight_qmax": str(WEIGHT_QMAX), "activation_qmax": str(ACTIVATION_QMAX)}

    def quantize_weight(self, name, weight, default_quantize=None):
        if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError(f"{name} must be an original floating-point matrix, got {weight.dtype} {tuple(weight.shape)}")
        values = weight.to(torch.float32, copy=True)
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} contains nonfinite values")
        scales = values.abs().amax(dim=1, keepdim=True).clamp_min_(1e-8).div_(WEIGHT_QMAX)
        quantized = values.div_(scales).clamp_(-WEIGHT_QMAX, WEIGHT_QMAX).to(torch.float8_e4m3fn)
        self.count += 1
        return quantized, scales, {}

    def validate(self):
        if not self.count:
            raise ValueError("No ProxyDINO linears were selected for FP8-F16 accumulation conversion")
