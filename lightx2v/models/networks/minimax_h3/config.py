from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_REMOVED_EXECUTION_OPTIONS = (
    "sgl_aligned",
    "h3_packed_sequence_alignment",
    "h3_rng_mode",
    "h3_step_update",
    "sglang_compatible_export",
    "rope_type",
    "keep_latents_dtype_in_scheduler",
)


@dataclass(frozen=True)
class MiniMaxH3ExecutionProfile:
    sampling_profile: str
    packed_sequence_alignment: int = 64


def resolve_minimax_h3_execution_profile(config: Mapping[str, Any]) -> MiniMaxH3ExecutionProfile:
    """Resolve the fixed H3 execution contract and its checkpoint sampling profile."""
    if "h3_sglang_parity_ops" in config:
        raise ValueError("MiniMax-H3 h3_sglang_parity_ops was removed; the aligned operators are always enabled")
    if "h3_ops" in config:
        raise ValueError("MiniMax-H3 h3_ops was removed; leaf backends use their standard registries")

    removed = [name for name in _REMOVED_EXECUTION_OPTIONS if name in config]
    if removed:
        names = ", ".join(removed)
        raise ValueError(
            f"MiniMax-H3 execution settings {names} were removed; aligned execution is the only implementation. "
            "Use h3_sampling_profile='base' or 'dmd' to select the checkpoint sampling contract."
        )

    sampling_profile = config.get("h3_sampling_profile", "base")
    if sampling_profile not in {"base", "dmd"}:
        raise ValueError(f"MiniMax-H3 h3_sampling_profile must be 'base' or 'dmd', got {sampling_profile!r}")
    return MiniMaxH3ExecutionProfile(sampling_profile=sampling_profile)


__all__ = [
    "MiniMaxH3ExecutionProfile",
    "resolve_minimax_h3_execution_profile",
]
