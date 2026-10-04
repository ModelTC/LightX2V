"""Trainable MiniMax-H3 building blocks used by LightX2V-Train."""

from .modeling import (
    init_empty_minimax_h3_transformer,
    load_minimax_h3_transformer,
    resolve_transformer_dir,
)
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
from .packing_ref2av import (
    MiniMaxH3ReferenceGeometry,
    build_ref2av_packed_sequence,
)
from .sequence_parallel import (
    MiniMaxH3SequenceParallelAttnProcessor,
    MiniMaxH3SequenceParallelInfo,
    install_minimax_h3_sequence_parallel,
)
from .sharded_loading import stream_load_minimax_h3_transformer
from .student_sla import (
    MiniMaxH3SparseOnlyAttnProcessor,
    MiniMaxH3StudentSLAConfig,
    install_minimax_h3_student_sla,
    retained_key_blocks,
)

__all__ = [
    "KEYFRAME_NOISE_AUG",
    "MiniMaxH3PackedSequence",
    "MiniMaxH3ReferenceGeometry",
    "MiniMaxH3SequenceParallelAttnProcessor",
    "MiniMaxH3SequenceParallelInfo",
    "MiniMaxH3SparseOnlyAttnProcessor",
    "MiniMaxH3StudentSLAConfig",
    "audio_latent_num_frames",
    "build_packed_sequence",
    "build_ref2av_packed_sequence",
    "build_row_timesteps",
    "keyframe_condition_noise",
    "init_empty_minimax_h3_transformer",
    "install_minimax_h3_student_sla",
    "install_minimax_h3_sequence_parallel",
    "load_minimax_h3_transformer",
    "patchify_video_latents",
    "retained_key_blocks",
    "resolve_transformer_dir",
    "stream_load_minimax_h3_transformer",
    "video_latent_num_frames",
]
