"""Shared metadata validation for FP8 checkpoints using FP16 accumulation."""

from pathlib import Path

from safetensors import safe_open


def validate_fp8_f16_accum_metadata(
    metadata,
    *,
    subject,
    quantization_profile,
    numeric_fields,
    profile_name="quantization_profile",
    include_actual_profile=False,
):
    """Check the model's profile and reduced-range numeric contract."""
    metadata = metadata or {}
    actual_profile = metadata.get("quantization_profile")
    if actual_profile != quantization_profile:
        if profile_name == "quantization_profile":
            message = f"{subject} requires {profile_name}={quantization_profile!r}"
        else:
            message = f"{subject} requires {profile_name} {quantization_profile!r}"
        if include_actual_profile:
            message += f", got {actual_profile!r}"
        raise ValueError(message)

    for name, expected in numeric_fields:
        try:
            actual = float(metadata.get(name))
        except (TypeError, ValueError):
            actual = None
        if actual != expected:
            raise ValueError(f"{subject} requires {name}={expected}, got {metadata.get(name)!r}")


def validate_fp8_f16_accum_checkpoint(
    checkpoint_path,
    *,
    quantization_profile,
    numeric_fields,
    not_found_prefix,
    profile_name="quantization_profile",
    include_actual_profile=False,
):
    """Validate one file or every safetensors shard in a checkpoint directory."""
    path = Path(checkpoint_path)
    files = (path,) if path.is_file() else tuple(sorted(path.glob("*.safetensors")))
    if not files:
        raise FileNotFoundError(f"{not_found_prefix}: {path}")

    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata()
        validate_fp8_f16_accum_metadata(
            metadata,
            subject=filename,
            quantization_profile=quantization_profile,
            numeric_fields=numeric_fields,
            profile_name=profile_name,
            include_actual_profile=include_actual_profile,
        )
