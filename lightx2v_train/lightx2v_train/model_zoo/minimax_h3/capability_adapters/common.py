"""Joint audio/video latent value objects used by MiniMax-H3."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class MiniMaxH3LatentShape:
    """Packed latent geometry for one H3 audio/video sample."""

    num_frames: int
    latent_frames: int
    latent_height: int
    latent_width: int
    audio_latents: int
    video_tokens: tuple[int, int, int]
    audio_tokens: tuple[int, int, int]


@dataclass(frozen=True)
class MiniMaxH3JointLatents:
    """Video and audio token tensors that form one H3 diffusion state."""

    video: Tensor
    audio: Tensor
    shape: MiniMaxH3LatentShape
