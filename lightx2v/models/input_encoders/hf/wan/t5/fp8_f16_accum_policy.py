"""Reduced-range UMT5 checkpoint contract for SM120 FP16 accumulation."""

from pathlib import Path

from safetensors import safe_open

QUANTIZATION_PROFILE = "wan-t5-fp8-f16-accum"
WEIGHT_QMAX = 14.0
ACTIVATION_QMAX = 7.0


def validate_checkpoint(checkpoint_path):
    path = Path(checkpoint_path)
    files = (path,) if path.is_file() else tuple(sorted(path.glob("*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No FP8-F16 T5 checkpoint found: {path}")
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
        if metadata.get("quantization_profile") != QUANTIZATION_PROFILE:
            raise ValueError(f"{filename} requires quantization_profile={QUANTIZATION_PROFILE!r}")
        for name, expected in (("weight_qmax", WEIGHT_QMAX), ("activation_qmax", ACTIVATION_QMAX)):
            try:
                value = float(metadata.get(name))
            except (TypeError, ValueError):
                value = None
            if value != expected:
                raise ValueError(f"{filename} requires {name}={expected}, got {metadata.get(name)!r}")
