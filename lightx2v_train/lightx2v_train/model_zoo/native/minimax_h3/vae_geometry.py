"""MiniMax-H3 video-VAE geometry shared by training and reconstruction."""

import math
from dataclasses import dataclass


VIDEO_FPS = 24
CLIP_LENGTH = 17
LATENTS_PER_CHUNK = 5
TOKEN_DROP = 3
TEMPORAL_COMPRESSION_RATIO = 4
SPATIAL_COMPRESSION_RATIO = 16
FRAME_PRE_PADDING = (-CLIP_LENGTH) % TEMPORAL_COMPRESSION_RATIO
TOKENS_CHUNK_SIZE = math.ceil(CLIP_LENGTH / TEMPORAL_COMPRESSION_RATIO)
TOKEN_OVERLAP = (-TOKEN_DROP) % TOKENS_CHUNK_SIZE
FRAME_OVERLAP = TOKEN_OVERLAP * TEMPORAL_COMPRESSION_RATIO - FRAME_PRE_PADDING
MIN_SOURCE_FRAMES = 4 * VIDEO_FPS
MAX_SOURCE_FRAMES = 15 * VIDEO_FPS


@dataclass(frozen=True)
class SpatialTileLayout:
    height_slices: tuple[slice, ...]
    width_slices: tuple[slice, ...]
    height_overlaps: tuple[int, ...]
    width_overlaps: tuple[int, ...]

    @property
    def num_tiles(self) -> int:
        return len(self.height_slices) * len(self.width_slices)


def align_num_frames(num_frames: int) -> int:
    """Pad a source clip to H3's ``17*n+5`` pixel-frame contract."""

    while num_frames % CLIP_LENGTH != LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    if num_frames % CLIP_LENGTH != LATENTS_PER_CHUNK:
        raise ValueError(f"MiniMax-H3 video frames must have the form 17*n+5, got {num_frames}.")
    return (num_frames - LATENTS_PER_CHUNK) // CLIP_LENGTH * LATENTS_PER_CHUNK + 2


def video_num_frames_from_latent(num_latent_frames: int) -> int:
    if (num_latent_frames - 2) % LATENTS_PER_CHUNK:
        raise ValueError(f"MiniMax-H3 latent frames must have the form 5*n+2, got {num_latent_frames}.")
    return (num_latent_frames - 2) // LATENTS_PER_CHUNK * CLIP_LENGTH + LATENTS_PER_CHUNK


def validate_source_num_frames(num_frames: int) -> None:
    maximum = align_num_frames(MAX_SOURCE_FRAMES)
    if not MIN_SOURCE_FRAMES <= num_frames <= maximum:
        raise ValueError(
            f"MiniMax-H3 VAE accepts 4-15 second source clips at {VIDEO_FPS} fps "
            f"({MIN_SOURCE_FRAMES}-{MAX_SOURCE_FRAMES} source frames, up to {maximum} when already aligned), got {num_frames}."
        )


def validate_aligned_num_frames(num_frames: int) -> None:
    video_latent_num_frames(num_frames)
    minimum = align_num_frames(MIN_SOURCE_FRAMES)
    maximum = align_num_frames(MAX_SOURCE_FRAMES)
    if not minimum <= num_frames <= maximum:
        raise ValueError(f"Aligned MiniMax-H3 clips must contain {minimum}-{maximum} frames, got {num_frames}.")


def temporal_decode_windows(num_latent_frames: int) -> tuple[tuple[int, int], ...]:
    """Return H3's overlapping seven-token raw-decoder windows."""

    video_num_frames_from_latent(num_latent_frames)
    num_tokens = num_latent_frames + TOKEN_DROP
    pad_tokens = (-num_tokens) % TOKENS_CHUNK_SIZE
    num_clips = (num_tokens + pad_tokens) // TOKENS_CHUNK_SIZE - int(TOKEN_DROP > 0)
    return tuple(
        (index * TOKENS_CHUNK_SIZE, index * TOKENS_CHUNK_SIZE + TOKENS_CHUNK_SIZE + TOKEN_OVERLAP)
        for index in range(num_clips)
    )


def split_tiles(length: int, tile_size: int, min_overlap: int, multiple: int) -> tuple[list[int], list[int], list[int]]:
    if tile_size >= length:
        return [0], [length], []

    num_tiles = math.ceil(length / tile_size)
    while tile_size * num_tiles - min_overlap * (num_tiles - 1) < length:
        num_tiles += 1

    overlaps = [min_overlap] * (num_tiles - 1)
    remaining = tile_size * num_tiles - sum(overlaps) - length
    for index in range(remaining // multiple):
        overlaps[index % (num_tiles - 1)] += multiple

    starts = [0]
    for index, overlap in enumerate(overlaps):
        starts.append(starts[index] + tile_size - overlap)
    return starts, [tile_size] * num_tiles, overlaps


def spatial_tile_layout(
    latent_height: int,
    latent_width: int,
    *,
    tile_height: int = 256,
    tile_width: int = 256,
    overlap_height: int = 64,
    overlap_width: int = 64,
) -> SpatialTileLayout:
    sample_height = latent_height * SPATIAL_COMPRESSION_RATIO
    sample_width = latent_width * SPATIAL_COMPRESSION_RATIO
    height_starts, height_lengths, height_overlaps = split_tiles(
        sample_height,
        tile_height,
        overlap_height,
        SPATIAL_COMPRESSION_RATIO,
    )
    width_starts, width_lengths, width_overlaps = split_tiles(
        sample_width,
        tile_width,
        overlap_width,
        SPATIAL_COMPRESSION_RATIO,
    )

    height_slices = tuple(
        slice(start // SPATIAL_COMPRESSION_RATIO, (start + length) // SPATIAL_COMPRESSION_RATIO)
        for start, length in zip(height_starts, height_lengths, strict=True)
    )
    width_slices = tuple(
        slice(start // SPATIAL_COMPRESSION_RATIO, (start + length) // SPATIAL_COMPRESSION_RATIO)
        for start, length in zip(width_starts, width_lengths, strict=True)
    )
    return SpatialTileLayout(
        height_slices,
        width_slices,
        tuple(height_overlaps),
        tuple(width_overlaps),
    )
