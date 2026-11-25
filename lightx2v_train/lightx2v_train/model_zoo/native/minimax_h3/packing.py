"""Base-task packed-sequence geometry for the trainable MiniMax-H3 DiT."""

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
KEYFRAME_NOISE_AUG = 0.999
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


def patchify_video_latents(
    latents: torch.Tensor,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> torch.Tensor:
    """Convert ``[1,C,F,H,W]`` keyframe/noise latents to H3 video rows."""
    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = latents.shape
    if batch != 1 or frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Cannot patchify latent shape {tuple(latents.shape)} with patch {patch_size}.")
    latents = latents.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    return latents.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def keyframe_condition_noise(
    condition_latent_shapes: tuple[tuple[int, int, int], ...],
    patch_size: tuple[int, int, int],
    latent_channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw and patchify one 5D noise tensor per keyframe, in packed order."""
    return torch.cat([patchify_video_latents(torch.randn((1, latent_channels, frames, height, width), device=device, dtype=dtype), patch_size) for frames, height, width in condition_latent_shapes])


def build_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    keyframe_anchors: tuple[str, ...] = (),
    reference_sizes: tuple[tuple[int, int], ...] = (),
) -> MiniMaxH3PackedSequence:
    """Pack text, fixed visual conditions, target audio, and target video."""
    if keyframe_anchors and reference_sizes:
        raise ValueError("Use either keyframes or Ref2AV images.")
    patch_t, patch_h, patch_w = patch_size
    if patch_t != 1:
        raise ValueError(f"MiniMax-H3 base tasks expect temporal patch size 1, got {patch_size}.")
    if latent_height % patch_h or latent_width % patch_w:
        raise ValueError(f"Latent canvas {latent_height}x{latent_width} is not divisible by patch {patch_size}.")
    text_token_tags = text_token_tags.to(device="cpu", dtype=torch.long).flatten()
    num_text = int(text_token_tags.numel())
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_condition_rows = len(keyframe_anchors) * rows_per_frame + sum((h // patch_h) * (w // patch_w) for h, w in reference_sizes)
    num_audio_rows = num_audio_latents * AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    condition_start = num_text
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows
    sequence_length = video_start + num_video_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float64)
    sqrt_area = float(np.sqrt(latent_height * latent_width))
    height_grid = _spatial_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_grid(latent_width, patch_w, sqrt_area)
    frame_grid = torch.stack([axis.reshape(-1) for axis in torch.meshgrid(height_grid, width_grid, indexing="ij")], dim=-1)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text)
        elif anchor == "last":
            anchor_time = float(num_text) + sum(_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % 5] for i in range(num_latent_frames)) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(
            condition_start + index * rows_per_frame,
            condition_start + (index + 1) * rows_per_frame,
        )
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    cursor = condition_start
    for index, (height, width) in enumerate(reference_sizes):
        area = float(np.sqrt(height * width))
        reference_grid = torch.cartesian_prod(_spatial_grid(height, patch_h, area), _spatial_grid(width, patch_w, area))
        rows = slice(cursor, cursor + len(reference_grid))
        position_ids[rows, 0] = num_text + index
        position_ids[rows, 1:] = reference_grid
        cursor = rows.stop

    target_time = float(num_text + len(reference_sizes))
    audio_time = target_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        (
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        )
    )
    video_positions = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_positions[:, :, 0] = _temporal_grid(num_latent_frames, target_time)[:, None]
    video_positions[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_positions.reshape(-1, 3)

    text_indices = torch.arange(num_text, dtype=torch.long)
    audio_indices = torch.arange(audio_start, video_start, dtype=torch.long)
    video_indices = torch.cat(
        (
            torch.arange(condition_start, audio_start, dtype=torch.long),
            torch.arange(video_start, sequence_length, dtype=torch.long),
        )
    )
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
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
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
    row_timesteps = torch.full((layout.sequence_length,), video_t, dtype=torch.float32, device=layout.position_ids.device)
    if layout.num_condition_video_rows:
        row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = max(video_t, KEYFRAME_NOISE_AUG)
    row_timesteps[layout.audio_indices] = audio_t
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)
