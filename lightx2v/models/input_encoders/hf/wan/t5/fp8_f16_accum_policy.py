"""Reduced-range UMT5 checkpoint contract for SM120 FP16 accumulation."""

from lightx2v.common.fp8_f16_accum_checkpoint import validate_fp8_f16_accum_checkpoint

QUANTIZATION_PROFILE = "wan-t5-fp8-f16-accum"
WEIGHT_QMAX = 14.0
ACTIVATION_QMAX = 7.0


def validate_checkpoint(checkpoint_path):
    validate_fp8_f16_accum_checkpoint(
        checkpoint_path,
        quantization_profile=QUANTIZATION_PROFILE,
        numeric_fields=(("weight_qmax", WEIGHT_QMAX), ("activation_qmax", ACTIVATION_QMAX)),
        not_found_prefix="No FP8-F16 T5 checkpoint found",
    )
