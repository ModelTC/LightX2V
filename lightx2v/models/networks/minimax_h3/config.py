from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_SGL_ALIGNED_PROFILE = {
    "h3_sglang_parity_ops": True,
    "h3_packed_sequence_alignment": 64,
    "h3_rng_mode": "sglang",
    "h3_step_update": "sglang_reference_blend",
    "sglang_compatible_export": True,
}

_LEGACY_DEFAULTS = {
    "h3_sglang_parity_ops": False,
    "h3_packed_sequence_alignment": 1,
    "h3_rng_mode": "legacy_stream",
    "h3_step_update": "reference_blend",
    "sglang_compatible_export": False,
}


@dataclass(frozen=True)
class MiniMaxH3SGLAlignment:
    parity_ops: bool
    packed_sequence_alignment: int
    rng_mode: str
    step_update: str
    compatible_export: bool


def resolve_minimax_h3_sgl_alignment(config: Mapping[str, Any]) -> MiniMaxH3SGLAlignment:
    """Resolve the atomic SGL-alignment profile without mutating ``config``.

    The legacy keys remain supported when ``sgl_aligned`` is absent. When the
    profile is explicitly selected, conflicting legacy values are rejected so
    that the single switch always represents a complete, atomic behavior set.
    """
    aligned = config.get("sgl_aligned", False)
    if type(aligned) is not bool:
        raise ValueError(f"MiniMax-H3 sgl_aligned must be true or false, got {aligned!r}")

    if aligned:
        conflicts = [
            f"{key}={config[key]!r} (expected {expected!r})"
            for key, expected in _SGL_ALIGNED_PROFILE.items()
            if key in config and config[key] != expected
        ]
        if conflicts:
            details = "; ".join(conflicts)
            raise ValueError(
                f"MiniMax-H3 sgl_aligned=True conflicts with legacy settings: {details}. "
                "Remove the legacy keys and let sgl_aligned control the profile."
            )
        resolved = _SGL_ALIGNED_PROFILE
    else:
        resolved = {key: config.get(key, default) for key, default in _LEGACY_DEFAULTS.items()}

    return MiniMaxH3SGLAlignment(
        parity_ops=bool(resolved["h3_sglang_parity_ops"]),
        packed_sequence_alignment=int(resolved["h3_packed_sequence_alignment"]),
        rng_mode=resolved["h3_rng_mode"],
        step_update=resolved["h3_step_update"],
        compatible_export=bool(resolved["sglang_compatible_export"]),
    )


__all__ = ["MiniMaxH3SGLAlignment", "resolve_minimax_h3_sgl_alignment"]
