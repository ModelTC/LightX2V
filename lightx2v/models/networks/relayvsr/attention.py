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


def get_local_attention_ops():
    if PLATFORM == "musa":
        from .musa_attention import musa_flash_attention, musa_spatial_attention

        return musa_flash_attention, musa_spatial_attention

    from .flash_attention import triton_local_attention
    from .wan_attention import spatial_attention

    return triton_local_attention, spatial_attention
