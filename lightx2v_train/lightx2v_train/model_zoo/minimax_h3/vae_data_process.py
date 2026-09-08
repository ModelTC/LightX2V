"""Video-only preprocessing for MiniMax-H3 VAE distillation."""

import math

import torch

from lightx2v_train.model_zoo.native.minimax_h3.vae_geometry import (
    SPATIAL_COMPRESSION_RATIO,
    VIDEO_FPS,
    align_num_frames,
    validate_aligned_num_frames,
    validate_source_num_frames,
    video_latent_num_frames,
)
from lightx2v_train.utils.registry import SAMPLE_PROCESSOR_REGISTER


class MiniMaxH3VAEDistillationProcessor:
    unconditional_prompt = " "
    requires_audio = False

    def __init__(self, *, load_cached_latents=True):
        self.load_cached_latents = load_cached_latents

    def __call__(self, sample):
        video = sample["inputs"]["video"]
        if not torch.is_tensor(video) or video.ndim != 4 or video.shape[0] != 3:
            shape = tuple(video.shape) if torch.is_tensor(video) else type(video).__name__
            raise ValueError(f"MiniMax-H3 VAE expects decoded video [3,F,H,W], got {shape}.")

        source_num_frames = int(video.shape[1])
        source_frame_rate = float(sample["meta"]["source_frame_rate"])
        if not math.isclose(source_frame_rate, VIDEO_FPS, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError(f"MiniMax-H3 VAE requires {VIDEO_FPS} fps video, got {source_frame_rate:g} fps.")
        validate_source_num_frames(source_num_frames)
        aligned_num_frames = align_num_frames(source_num_frames)
        validate_aligned_num_frames(aligned_num_frames)
        if aligned_num_frames > source_num_frames:
            video = torch.cat(
                (video, video[:, -1:].repeat(1, aligned_num_frames - source_num_frames, 1, 1)),
                dim=1,
            )

        height, width = video.shape[-2:]
        if height % 32 or width % 32:
            raise ValueError(f"MiniMax-H3 VAE height and width must be divisible by 32, got {height}x{width}.")

        # VideoDataset decodes to [-1, 1]; H3's ImageNet preprocessing starts at [0, 1].
        sample["inputs"]["video"] = ((video + 1.0) * 0.5).clamp_(0.0, 1.0)
        latent_path = sample["meta"].get("latent_path")
        if self.load_cached_latents and latent_path is not None:
            latent = torch.load(latent_path, map_location="cpu", weights_only=True)
            if isinstance(latent, dict):
                if not latent.get("normalized", False):
                    raise ValueError(f"Cached MiniMax-H3 latent is not normalized: {latent_path}")
                latent = latent.get("latents")
            expected_shape = (
                24,
                video_latent_num_frames(aligned_num_frames),
                height // SPATIAL_COMPRESSION_RATIO,
                width // SPATIAL_COMPRESSION_RATIO,
            )
            if not torch.is_tensor(latent) or tuple(latent.shape) != expected_shape:
                shape = tuple(latent.shape) if torch.is_tensor(latent) else type(latent).__name__
                raise ValueError(f"Cached MiniMax-H3 latent must have shape {expected_shape}, got {shape}: {latent_path}")
            sample["inputs"]["latents"] = latent
        sample["meta"].update(
            {
                "source_num_frames": source_num_frames,
                "num_frames": aligned_num_frames,
                "target_height": int(height),
                "target_width": int(width),
            }
        )
        return sample


# Older pruned-decoder checkpoint configs use this processor alias (not a CNN model).
@SAMPLE_PROCESSOR_REGISTER("minimax_h3_turbo_vae")
@SAMPLE_PROCESSOR_REGISTER("minimax_h3_pruned_vae")
def build_minimax_h3_vae_processor(config):
    del config
    return MiniMaxH3VAEDistillationProcessor()


@SAMPLE_PROCESSOR_REGISTER("minimax_h3_pruned_encoder")
def build_minimax_h3_encoder_processor(config):
    del config
    # Cached mode latents contain neither posterior variance nor an encoder graph.
    return MiniMaxH3VAEDistillationProcessor(load_cached_latents=False)
