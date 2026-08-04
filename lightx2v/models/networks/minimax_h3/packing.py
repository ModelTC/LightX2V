"""MiniMax-H3 base-task packed-sequence geometry.

This is a small, runtime-independent port of the released MiniMax-H3 packing
contract.  The transformer consumes one unbatched sequence ordered as
``[text | keyframes | stereo audio | video]``.  Keeping this code next to the native
network makes the row order, rotary clock, and noise layout explicit.
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2

FPS = 24
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
CANVAS_MULTIPLE = 32
SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
MIN_ASPECT_RATIO = 1.0 / 4.0
MAX_ASPECT_RATIO = 4.0
MIN_DURATION = 5.0
MAX_DURATION = 15.0
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
KEYFRAME_NOISE_AUG = 0.999
KEYFRAME_ENCODE_SEED = 42

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


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """Resolve an aspect ratio to the released 768p/area-capped H3 canvas."""
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}")
    ratio = aspect_width / aspect_height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {aspect_width}:{aspect_height}")
    if ratio >= 1.0:
        width, height = SHORT_EDGE * ratio, float(SHORT_EDGE)
    else:
        width, height = float(SHORT_EDGE), SHORT_EDGE / ratio
    area = width * height
    if area > MAX_PIXELS:
        scale = (MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale
    return (
        max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def prepare_keyframe_image(image: Image.Image, height: int, width: int, stretch: bool) -> Image.Image:
    """Stretch the geometry anchor or cover-crop a following keyframe."""
    if image.size == (width, height):
        return image
    if stretch:
        return image.resize((width, height), Image.Resampling.LANCZOS)
    scale = max(width / image.size[0], height / image.size[1])
    resized_size = (max(width, round(image.size[0] * scale)), max(height, round(image.size[1] * scale)))
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    return image.resize(resized_size, Image.Resampling.LANCZOS).crop((left, top, left + width, top + height))


def align_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        raise ValueError(f"target_video_length must be positive, got {num_frames}")
    while num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    if num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(f"MiniMax-H3 frame count must have the form 17*n+5, got {num_frames}")
    return (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def validate_t2av_geometry(num_frames: int, height: int, width: int) -> None:
    min_aligned_frames = align_num_frames(round(MIN_DURATION * FPS))
    max_aligned_frames = align_num_frames(round(MAX_DURATION * FPS))
    if not min_aligned_frames <= num_frames <= max_aligned_frames:
        raise ValueError(
            f"MiniMax-H3 supports aligned frame counts from {min_aligned_frames} to {max_aligned_frames} ({MIN_DURATION:g}-{MAX_DURATION:g}s requested at {FPS} fps); got {num_frames} frames"
        )
    if height % CANVAS_MULTIPLE or width % CANVAS_MULTIPLE:
        raise ValueError(f"MiniMax-H3 height and width must be multiples of {CANVAS_MULTIPLE}, got {height}x{width}")
    video_latent_num_frames(num_frames)


def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int] = (1, 2, 2)) -> torch.Tensor:
    """Convert ``[1,C,F,H,W]`` latents to unbatched frame-major rows."""
    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = latents.shape
    if batch != 1:
        raise ValueError(f"T2AV currently supports batch size 1, got {batch}")
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"latent shape {tuple(latents.shape)} is not divisible by patch {patch_size}")
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
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_tokens(
    rows: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int = 24,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> torch.Tensor:
    patch_t, patch_h, patch_w = patch_size
    rows = rows.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(-1, channels, num_latent_frames, latent_height, latent_width).contiguous()


def unpack_audio_tokens(rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
    rows = rows.reshape(AUDIO_CHANNELS, num_audio_latents, rows.shape[-1])
    return rows.permute(0, 2, 1).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # NumPy's endpoint=False arithmetic is part of the published checkpoint
    # contract and differs by an ulp from torch.linspace in some shapes.
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)] for i in range(num_latent_frames)],
        dtype=torch.float64,
    )
    return origin + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))


def temporal_position_span(num_latent_frames: int) -> float:
    """Exact NumPy-summed rotary span used for last-frame/reference anchors."""
    spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for index, multiplier in enumerate(_ROPE_FRAMES_PER_LATENT):
        spans[index :: len(_ROPE_FRAMES_PER_LATENT)] *= multiplier
    return float(spans.sum())


def build_t2av_packed_sequence(
    num_text_tokens: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> MiniMaxH3PackedSequence:
    """Build the exact padless ``[text | audio | video]`` T2AV layout."""
    return build_packed_sequence(
        torch.full((num_text_tokens,), TEXT_TAG, dtype=torch.long),
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch_size,
    )


def build_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    keyframe_anchors: tuple[str, ...] = (),
) -> MiniMaxH3PackedSequence:
    """Build ``[Qwen rows | keyframe rows | target audio | target video]``."""
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = int(text_token_tags.shape[0])
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows
    sequence_length = video_start + num_video_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    frame_grid = torch.stack([grid.reshape(-1) for grid in torch.meshgrid(height_grid, width_grid, indexing="ij")], dim=-1)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            anchor_time = float(num_text_tokens) + temporal_position_span(num_latent_frames) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        (
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        )
    )

    video_positions = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_positions[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_positions[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_positions.reshape(-1, 3)

    text_indices = torch.arange(num_text_tokens, dtype=torch.long)
    audio_indices = torch.arange(audio_start, video_start, dtype=torch.long)
    video_indices = torch.cat((torch.arange(condition_start, audio_start, dtype=torch.long), torch.arange(video_start, sequence_length, dtype=torch.long)))
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG

    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
    )


def build_row_timesteps(
    layout: MiniMaxH3PackedSequence,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float | None = None,
    condition_audio_timestep: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_timesteps = torch.full((layout.sequence_length,), video_timestep, dtype=torch.float32)
    if condition_video_timestep is None:
        condition_video_timestep = max(video_timestep, KEYFRAME_NOISE_AUG)
    row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = condition_video_timestep
    row_timesteps[layout.audio_indices[layout.num_condition_audio_rows :]] = audio_timestep
    row_timesteps[layout.audio_indices[: layout.num_condition_audio_rows]] = condition_audio_timestep
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


def keyframe_condition_noise(
    condition_latent_shapes: tuple[tuple[int, int, int], ...],
    patch_size: tuple[int, int, int],
    latent_channels: int,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw one CPU condition-noise tensor per visual condition, in request order."""
    rows = []
    for num_frames, height, width in condition_latent_shapes:
        noise = torch.randn(
            (1, latent_channels, num_frames, height, width),
            generator=generator,
            device="cpu",
            dtype=dtype,
        )
        rows.append(patchify_video_latents(noise, patch_size))
    if not rows:
        return torch.empty((0, latent_channels * int(np.prod(patch_size))), dtype=dtype)
    return torch.cat(rows)
