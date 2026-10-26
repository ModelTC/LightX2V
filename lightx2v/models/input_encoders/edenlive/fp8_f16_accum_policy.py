"""Reduced-range ProxyDINO checkpoint contract for SM120 FP16 accumulation."""

from safetensors import safe_open

QUANTIZATION_PROFILE = "edenlive-proxy-fp8-f16-accum"
WEIGHT_QMAX = 14.0
ACTIVATION_QMAX = 7.0


def validate_checkpoint(checkpoint_path):
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    if metadata.get("quantization_profile") != QUANTIZATION_PROFILE:
        raise ValueError(f"ProxyDINO FP8-F16 checkpoint requires quantization_profile={QUANTIZATION_PROFILE!r}")
    for name, expected in (("weight_qmax", WEIGHT_QMAX), ("activation_qmax", ACTIVATION_QMAX)):
        try:
            actual = float(metadata.get(name))
        except (TypeError, ValueError):
            actual = None
        if actual != expected:
            raise ValueError(f"ProxyDINO FP8-F16 checkpoint requires {name}={expected}, got {metadata.get(name)!r}")
