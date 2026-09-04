"""MiniMax-H3 checkpoint policy for FP8 GEMM with FP16 accumulation."""

import torch

from lightx2v.models.networks.minimax_h3.fp8_f16_accum_policy import (
    FP8_F16_ACCUM_PROJECTION_SUFFIXES,
    FP8_F16_ACCUM_QUANTIZATION_PROFILE,
    FP8_F16_ACCUM_WEIGHT_QMAX,
)

# DiT has 50 x (six main projections + AdaLN); VAE has 36 x six projections + proj_out.
_EXPECTED_QUANTIZED_COUNTS = {
    "h3": 350,
    "h3_video_vae_decoder": 217,
}
_EXPECTED_QMAX14_COUNTS = {
    "h3": 300,
    "h3_video_vae_decoder": 216,
}


class H3FP8F16AccumQuantization:
    """Assign qmax14 only to H3 projections using FP16 accumulation."""

    def __init__(self, model_type):
        if model_type not in _EXPECTED_QUANTIZED_COUNTS:
            raise ValueError(f"{FP8_F16_ACCUM_QUANTIZATION_PROFILE} does not support model_type={model_type!r}")
        self.model_type = model_type
        self.quantized_count = 0
        self.qmax14_count = 0

    @property
    def metadata(self):
        return {
            "format": "pt",
            "quantization_profile": FP8_F16_ACCUM_QUANTIZATION_PROFILE,
            "weight_qmax": str(FP8_F16_ACCUM_WEIGHT_QMAX),
        }

    def quantize_weight(self, name, weight, default_quantize):
        projection_name = name.removesuffix(".weight")
        uses_reduced_range = projection_name.endswith(FP8_F16_ACCUM_PROJECTION_SUFFIXES)
        self.quantized_count += 1
        if not uses_reduced_range:
            return default_quantize(weight)

        values = weight.float()
        scales = values.abs().amax(dim=1, keepdim=True).clamp_min_(1e-8).div_(FP8_F16_ACCUM_WEIGHT_QMAX)
        values.div_(scales).clamp_(-FP8_F16_ACCUM_WEIGHT_QMAX, FP8_F16_ACCUM_WEIGHT_QMAX)

        self.qmax14_count += 1
        return values.to(torch.float8_e4m3fn), scales, {}

    def validate(self):
        expected_quantized = _EXPECTED_QUANTIZED_COUNTS[self.model_type]
        expected_qmax14 = _EXPECTED_QMAX14_COUNTS[self.model_type]
        if self.quantized_count != expected_quantized or self.qmax14_count != expected_qmax14:
            raise ValueError(f"Unexpected {self.model_type} FP8 conversion coverage: quantized={self.quantized_count}/{expected_quantized}, qmax14={self.qmax14_count}/{expected_qmax14}")


def create_h3_fp8_f16_accum_quantization(profile, model_type):
    if profile is None:
        return None
    if profile != FP8_F16_ACCUM_QUANTIZATION_PROFILE:
        raise ValueError(f"Unsupported quantization profile: {profile}")
    return H3FP8F16AccumQuantization(model_type)
