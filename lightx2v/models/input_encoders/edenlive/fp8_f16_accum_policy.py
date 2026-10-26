"""Reduced-range ProxyDINO checkpoint contract for SM120 FP16 accumulation."""

from safetensors import safe_open

from lightx2v.common.fp8_f16_accum_checkpoint import validate_fp8_f16_accum_metadata

QUANTIZATION_PROFILE = "edenlive-proxy-fp8-f16-accum"
WEIGHT_QMAX = 14.0
ACTIVATION_QMAX = 7.0


def validate_checkpoint(checkpoint_path):
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
    validate_fp8_f16_accum_metadata(
        metadata,
        subject="ProxyDINO FP8-F16 checkpoint",
        quantization_profile=QUANTIZATION_PROFILE,
        numeric_fields=(("weight_qmax", WEIGHT_QMAX), ("activation_qmax", ACTIVATION_QMAX)),
    )
