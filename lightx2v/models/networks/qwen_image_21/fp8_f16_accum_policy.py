"""Reduced-range FP8 checkpoint contract for Qwen-Image-2.1 FP16 accumulation."""

from lightx2v.utils.fp8_f16_accum_checkpoint import validate_fp8_f16_accum_checkpoint

QUANTIZATION_PROFILE = "qwen-image-21-fp8-f16-accum"
WEIGHT_QMAX = 14.0
ACTIVATION_QMAX = 7.0


def validate_checkpoint(checkpoint_path):
    validate_fp8_f16_accum_checkpoint(
        checkpoint_path,
        quantization_profile=QUANTIZATION_PROFILE,
        numeric_fields=(("weight_qmax", WEIGHT_QMAX),),
        not_found_prefix="No safetensors checkpoint found",
    )
