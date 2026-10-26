"""RelayVSR attention geometry and platform-specific backend loading."""

from dataclasses import dataclass

from lightx2v_platform.base.global_var import PLATFORM


@dataclass(frozen=True)
class SpatialAttentionSpec:
    frames: int
    height: int
    width: int
    window_height: int
    window_width: int
    temporal_window: int
    key_frames: int
    query_frame_offset: int = 0
    key_frame_offset: int = 0


def resolve_local_attention_backend(deployment):
    backend = "torch_sdpa" if PLATFORM == "musa" else "triton"
    requested = deployment.get("local_attention_backend", "auto")
    if requested not in ("auto", backend):
        raise ValueError(f"RelayVSR on {PLATFORM} requires local_attention_backend={backend!r}; got {requested!r}")
    return backend


def get_local_attention_ops(deployment):
    if resolve_local_attention_backend(deployment) == "torch_sdpa":
        from .musa_attention import musa_flash_attention, musa_spatial_attention

        return musa_flash_attention, musa_spatial_attention

    from .flash_attention import triton_local_attention
    from .wan_attention import spatial_attention

    return triton_local_attention, spatial_attention
