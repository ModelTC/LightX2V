"""Reduced-range FP8 checkpoint contract for Wan FP16 accumulation."""

from pathlib import Path

from safetensors import safe_open

QUANTIZATION_PROFILE = "wan-ar-fp8-f16-accum"
WEIGHT_QMAX = 14.0
ACTIVATION_QMAX = 7.0


def validate_checkpoint(checkpoint_path):
    path = Path(checkpoint_path)
    files = (path,) if path.is_file() else tuple(sorted(path.glob("*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No FP8-F16 Wan checkpoint found: {path}")
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
        if metadata.get("quantization_profile") != QUANTIZATION_PROFILE:
            raise ValueError(f"{filename} requires quantization_profile={QUANTIZATION_PROFILE!r}")
        try:
            weight_qmax = float(metadata.get("weight_qmax"))
        except (TypeError, ValueError):
            weight_qmax = None
        if weight_qmax != WEIGHT_QMAX:
            raise ValueError(f"{filename} requires weight_qmax={WEIGHT_QMAX}, got {metadata.get('weight_qmax')!r}")
