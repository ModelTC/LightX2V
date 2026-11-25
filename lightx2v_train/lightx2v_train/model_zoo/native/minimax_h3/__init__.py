"""Trainable MiniMax-H3 building blocks used by LightX2V-Train."""

from .modeling import load_minimax_h3_transformer, resolve_transformer_dir
from .packing import (
    KEYFRAME_NOISE_AUG,
    MiniMaxH3PackedSequence,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
    video_latent_num_frames,
)

__all__ = [
    "KEYFRAME_NOISE_AUG",
    "MiniMaxH3PackedSequence",
    "audio_latent_num_frames",
    "build_packed_sequence",
    "build_row_timesteps",
    "keyframe_condition_noise",
    "load_minimax_h3_transformer",
    "patchify_video_latents",
    "resolve_transformer_dir",
    "video_latent_num_frames",
]
