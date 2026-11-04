"""Checkpoint contracts for reduced-range FP8 weights with FP16 accumulation."""

from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open

WEIGHT_QMAX = 14.0
ACTIVATION_QMAX = 7.0
H3_VIDEO_VAE_ACTIVATION_QMAX = 14.0

WAN_QUANTIZATION_PROFILE = "wan-ar-fp8-f16-accum"
T5_QUANTIZATION_PROFILE = "wan-t5-fp8-f16-accum"
QWEN_IMAGE_21_QUANTIZATION_PROFILE = "qwen-image-21-fp8-f16-accum"
H3_QUANTIZATION_PROFILE = "h3-fp8-f16-accum"
EDENLIVE_PROXY_QUANTIZATION_PROFILE = "edenlive-proxy-fp8-f16-accum"

H3_PROJECTION_SUFFIXES = (
    ".attn.to_q",
    ".attn.to_k",
    ".attn.to_v",
    # The runtime-only fused projection uses the same accumulation contract.
    ".attn.to_qkv",
    ".attn.to_out.0",
    ".ff.net.0.proj",
    ".ff.net.2",
)


@dataclass(frozen=True)
class _CheckpointContract:
    numeric_fields: tuple[tuple[str, float], ...]
    not_found_prefix: str = ""
    single_file_subject: str | None = None
    profile_name: str = "quantization_profile"
    include_actual_profile: bool = False


_CONTRACTS = {
    WAN_QUANTIZATION_PROFILE: _CheckpointContract((("weight_qmax", WEIGHT_QMAX),), "No FP8-F16 Wan checkpoint found"),
    T5_QUANTIZATION_PROFILE: _CheckpointContract(
        (("weight_qmax", WEIGHT_QMAX), ("activation_qmax", ACTIVATION_QMAX)),
        "No FP8-F16 T5 checkpoint found",
    ),
    QWEN_IMAGE_21_QUANTIZATION_PROFILE: _CheckpointContract((("weight_qmax", WEIGHT_QMAX),), "No safetensors checkpoint found"),
    H3_QUANTIZATION_PROFILE: _CheckpointContract(
        (("weight_qmax", WEIGHT_QMAX),),
        "No safetensors weights found in FP8 checkpoint",
        profile_name="quantization profile",
        include_actual_profile=True,
    ),
    EDENLIVE_PROXY_QUANTIZATION_PROFILE: _CheckpointContract(
        (("weight_qmax", WEIGHT_QMAX), ("activation_qmax", ACTIVATION_QMAX)),
        single_file_subject="ProxyDINO FP8-F16 checkpoint",
    ),
}


def validate_checkpoint(checkpoint_path, *, profile):
    """Validate the checkpoint metadata for one supported model profile."""
    try:
        contract = _CONTRACTS[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown FP8-F16 checkpoint profile: {profile!r}") from exc

    if contract.single_file_subject is not None:
        # ProxyDINO checkpoints are single files and retain their public error text.
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata()
        validate_fp8_f16_accum_metadata(
            metadata,
            subject=contract.single_file_subject,
            quantization_profile=profile,
            numeric_fields=contract.numeric_fields,
        )
        return

    validate_fp8_f16_accum_checkpoint(
        checkpoint_path,
        quantization_profile=profile,
        numeric_fields=contract.numeric_fields,
        not_found_prefix=contract.not_found_prefix,
        profile_name=contract.profile_name,
        include_actual_profile=contract.include_actual_profile,
    )


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
