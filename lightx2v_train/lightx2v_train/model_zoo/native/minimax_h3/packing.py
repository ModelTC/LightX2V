"""T2AV packed-sequence geometry for the trainable MiniMax-H3 DiT."""

from dataclasses import dataclass

import numpy as np
import torch

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
FPS = 24
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32.0


@dataclass(frozen=True)
class MiniMaxH3PackedSequence:
    sequence_length: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int = 0
    num_condition_audio_rows: int = 0

    def to(self, device):
        return MiniMaxH3PackedSequence(
            sequence_length=self.sequence_length,
            position_ids=self.position_ids.to(device),
            token_tags=self.token_tags.to(device),
            video_indices=self.video_indices.to(device),
            audio_indices=self.audio_indices.to(device),
            text_indices=self.text_indices.to(device),
            num_condition_video_rows=self.num_condition_video_rows,
            num_condition_audio_rows=self.num_condition_audio_rows,
        )


def video_latent_num_frames(num_frames: int) -> int:
    if num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(f"MiniMax-H3 num_frames must be 17*n+5, got {num_frames}.")
    return (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def _spatial_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    values = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(values).to(torch.float64)


def _temporal_grid(num_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % 5] for i in range(num_frames)],
        dtype=torch.float64,
    )
    return origin + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))


def build_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> MiniMaxH3PackedSequence:
    """Build the padless T2AV layout ``[text | stereo audio | video]``."""
    patch_t, patch_h, patch_w = patch_size
    if patch_t != 1:
        raise ValueError(f"MiniMax-H3 T2AV expects temporal patch size 1, got {patch_size}.")
    if latent_height % patch_h or latent_width % patch_w:
        raise ValueError(f"Latent canvas {latent_height}x{latent_width} is not divisible by patch {patch_size}.")
    text_token_tags = text_token_tags.to(device="cpu", dtype=torch.long).flatten()
    num_text = int(text_token_tags.numel())
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_audio_rows = num_audio_latents * AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    audio_start = num_text
    video_start = audio_start + num_audio_rows
    sequence_length = video_start + num_video_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float64)
    sqrt_area = float(np.sqrt(latent_height * latent_width))
    height_grid = _spatial_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_grid(latent_width, patch_w, sqrt_area)
    frame_grid = torch.stack([axis.reshape(-1) for axis in torch.meshgrid(height_grid, width_grid, indexing="ij")], dim=-1)

    audio_time = float(num_text) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        (
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        )
    )
    video_positions = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_positions[:, :, 0] = _temporal_grid(num_latent_frames, float(num_text))[:, None]
    video_positions[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_positions.reshape(-1, 3)

    text_indices = torch.arange(num_text, dtype=torch.long)
    audio_indices = torch.arange(audio_start, video_start, dtype=torch.long)
    video_indices = torch.arange(video_start, sequence_length, dtype=torch.long)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG
    return MiniMaxH3PackedSequence(
        sequence_length,
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
    )


def build_row_timesteps(
    layout: MiniMaxH3PackedSequence,
    video_sigma: float | torch.Tensor,
    audio_sigma: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert modality sigmas to H3's clean-ward ``t=1-sigma`` rows."""
    # Keep the subtraction in float32. The published scheduler materializes
    # ``timesteps = 1 - sigmas`` as a tensor before converting values to
    # Python floats; doing the subtraction in Python differs by one ulp.
    video_t = float((1.0 - torch.as_tensor(video_sigma, dtype=torch.float32)).item())
    audio_t = float((1.0 - torch.as_tensor(audio_sigma, dtype=torch.float32)).item())
    row_timesteps = torch.full((layout.sequence_length,), video_t, dtype=torch.float32)
    row_timesteps[layout.audio_indices] = audio_t
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)
