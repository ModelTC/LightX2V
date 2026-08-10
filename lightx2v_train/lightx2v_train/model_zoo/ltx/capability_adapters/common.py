"""Shared latent value objects for LTX training capabilities."""

from dataclasses import dataclass

from torch import Tensor


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
