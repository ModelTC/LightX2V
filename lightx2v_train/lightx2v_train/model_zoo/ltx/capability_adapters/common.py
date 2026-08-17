"""Shared latent value objects for LTX training capabilities."""

from dataclasses import dataclass

import torch
from torch import Tensor

from lightx2v_train.model_zoo.native.ltx2 import (
    AudioLatentShape,
    AudioPatchifier,
    SpatioTemporalScaleFactors,
    VideoLatentPatchifier,
    VideoLatentShape,
    get_pixel_coords,
)


@dataclass(frozen=True)
class LTXJointLatentShape:
    video_latent: tuple[int, ...]
    audio_latent: tuple[int, ...]
    video_tokens: tuple[int, ...]
    audio_tokens: tuple[int, ...]
    fps: float


@dataclass(frozen=True)
class LTXJointLatents:
    video: Tensor
    audio: Tensor
    shape: LTXJointLatentShape


class LTXLatentCodec:
    """Validate, patchify, position, and flatten joint LTX latent tensors."""

    def __init__(self, model, *, default_fps: float = 24.0) -> None:
        self.model = model
        self.default_fps = float(default_fps)
        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)

    def prepare(self, inputs, dtype):
        video_data = inputs.get("video_latents")
        audio_data = inputs.get("audio_latents")
        if video_data is None or audio_data is None:
            raise KeyError("LTX training expects inputs.video_latents and inputs.audio_latents.")
        video_latents = self.video_latents(video_data, dtype)
        audio_latents = self.audio_latents(audio_data, dtype)
        video_tokens = self.video_patchifier.patchify(video_latents)
        audio_tokens = self.audio_patchifier.patchify(audio_latents)
        return video_data, audio_data, video_latents, audio_latents, video_tokens, audio_tokens

    def positions(self, video_data, video_latents, audio_tokens):
        video_positions = self.video_positions(video_data, video_latents)
        audio_positions = self.audio_patchifier.get_patch_grid_bounds(
            output_shape=AudioLatentShape(
                frames=audio_tokens.shape[1],
                mel_bins=16,
                batch=1,
                channels=8,
            ),
            device=audio_tokens.device,
        )
        return video_positions, audio_positions

    def flatten(self, video_tokens: Tensor, audio_tokens: Tensor) -> Tensor:
        if video_tokens.shape[0] != 1 or audio_tokens.shape[0] != 1:
            raise ValueError("LTX consistency training only supports physical batch size 1.")
        return torch.cat((video_tokens.flatten(1), audio_tokens.flatten(1)), dim=1)

    @staticmethod
    def unflatten(flat: Tensor, video_shape, audio_shape) -> tuple[Tensor, Tensor]:
        video_numel = int(torch.Size(video_shape).numel())
        audio_numel = int(torch.Size(audio_shape).numel())
        if flat.shape != (1, video_numel + audio_numel):
            raise ValueError(f"Flattened LTX latent must have shape {(1, video_numel + audio_numel)}, got {tuple(flat.shape)}.")
        video, audio = flat.split((video_numel, audio_numel), dim=1)
        return video.reshape(video_shape), audio.reshape(audio_shape)

    def video_latents(self, data, dtype):
        latents = data["latents"].to(device=self.model.device, dtype=dtype)
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        if latents.ndim != 5:
            raise ValueError(f"LTX video latents must have shape [B,C,F,H,W] or [C,F,H,W], got {tuple(latents.shape)}.")
        if latents.shape[0] != 1:
            raise ValueError("LTX training only supports physical batch size 1.")
        return latents.contiguous()

    def audio_latents(self, data, dtype):
        latents = data["latents"].to(device=self.model.device, dtype=dtype)
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)
        if latents.ndim != 4:
            raise ValueError(f"LTX audio latents must have shape [B,C,T,F] or [C,T,F], got {tuple(latents.shape)}.")
        if latents.shape[0] != 1:
            raise ValueError("LTX training only supports physical batch size 1.")
        return latents.contiguous()

    def video_positions(self, data, latents):
        frames = int(data["num_frames"][0].item()) if "num_frames" in data else latents.shape[2]
        height = int(data["height"][0].item()) if "height" in data else latents.shape[3]
        width = int(data["width"][0].item()) if "width" in data else latents.shape[4]
        fps = float(data["fps"][0].item()) if "fps" in data else self.default_fps
        latent_coords = self.video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                frames=frames,
                height=height,
                width=width,
                batch=1,
                channels=128,
            ),
            device=latents.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
        ).float()
        positions[:, 0, ...] /= fps
        return positions
