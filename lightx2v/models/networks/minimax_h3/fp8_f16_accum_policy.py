from pathlib import Path

from safetensors import safe_open

FP8_F16_ACCUM_WEIGHT_QMAX = 14.0
DIT_FP8_F16_ACCUM_ACTIVATION_QMAX = 7.0
VIDEO_VAE_FP8_F16_ACCUM_ACTIVATION_QMAX = 14.0
FP8_F16_ACCUM_QUANTIZATION_PROFILE = "h3-fp8-f16-accum"
FP8_F16_ACCUM_PROJECTION_SUFFIXES = (
    ".attn.to_q",
    ".attn.to_k",
    ".attn.to_v",
    ".attn.to_out.0",
    ".ff.net.0.proj",
    ".ff.net.2",
)


def validate_fp8_f16_accum_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    files = (checkpoint_path,) if checkpoint_path.is_file() else tuple(sorted(checkpoint_path.glob("*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors weights found in FP8 checkpoint: {checkpoint_path}")

    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
        profile = metadata.get("quantization_profile")
        if profile != FP8_F16_ACCUM_QUANTIZATION_PROFILE:
            raise ValueError(f"{filename} requires quantization profile {FP8_F16_ACCUM_QUANTIZATION_PROFILE!r}, got {profile!r}")
        try:
            weight_qmax = float(metadata.get("weight_qmax"))
        except (TypeError, ValueError):
            weight_qmax = None
        if weight_qmax != FP8_F16_ACCUM_WEIGHT_QMAX:
            raise ValueError(f"{filename} requires weight_qmax={FP8_F16_ACCUM_WEIGHT_QMAX}, got {metadata.get('weight_qmax')!r}")
