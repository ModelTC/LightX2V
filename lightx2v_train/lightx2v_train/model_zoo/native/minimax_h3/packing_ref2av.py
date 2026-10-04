"""Omni-reference packed-sequence geometry for MiniMax-H3 Ref2AV."""

from dataclasses import dataclass

import numpy as np
import torch

from .packing import (
    AUDIO_CHANNELS,
    AUDIO_TAG,
    TEXT_TAG,
    VIDEO_TAG,
    _ROPE_FRAMES_PER_LATENT,
    _ROPE_FRAME_RESCALE,
    MiniMaxH3PackedSequence,
    _spatial_grid,
    _temporal_grid,
)


@dataclass(frozen=True)
class MiniMaxH3ReferenceGeometry:
    """Latent geometry of one cached reference, in semantic request order."""

    kind: str
    num_latent_frames: int = 0
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0

    @property
    def has_audio(self) -> bool:
        return self.num_audio_latents > 0

    @property
    def num_video_rows(self) -> int:
        if self.kind == "audio":
            return 0
        return self.num_latent_frames * (self.latent_height // 2) * (self.latent_width // 2)

    @property
    def num_audio_rows(self) -> int:
        return self.num_audio_latents * AUDIO_CHANNELS


def _reference_temporal_position_span(num_latent_frames: int) -> float:
    """Advance a reference video's clock using the official sequential sum."""
    return sum(_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)] for index in range(num_latent_frames))


def _frame_position_grid(
    latent_height: int,
    latent_width: int,
    patch_h: int,
    patch_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_area = float(np.sqrt(latent_height * latent_width))
    height_grid = _spatial_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_grid(latent_width, patch_w, sqrt_area)
    frame_grid = torch.stack(
        [axis.reshape(-1) for axis in torch.meshgrid(height_grid, width_grid, indexing="ij")],
        dim=-1,
    )
    return frame_grid, width_grid


def _fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
) -> None:
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat(
        (
            torch.full(
                (num_audio_latents,),
                float(width_grid[0]),
                dtype=torch.float64,
            ),
            torch.full(
                (num_audio_latents,),
                float(width_grid[-1]),
                dtype=torch.float64,
            ),
        )
    )


def build_ref2av_packed_sequence(
    text_token_tags: torch.Tensor,
    references: tuple[MiniMaxH3ReferenceGeometry, ...] | list[MiniMaxH3ReferenceGeometry],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> MiniMaxH3PackedSequence:
    """Build ``[text | ordered reference blocks | target audio | target video]``.

    A video reference packs its audio immediately before its visual rows. Each
    reference advances the shared audio/video rotary clock, exactly matching
    Diffusers' ``build_ref2va_packed_sequence``.
    """
    _, patch_h, patch_w = patch_size
    if tuple(patch_size) != (1, 2, 2):
        raise ValueError(f"MiniMax-H3 Ref2AV expects patch size (1, 2, 2), got {patch_size}.")
    if latent_height % patch_h or latent_width % patch_w:
        raise ValueError(f"Target latent canvas {latent_height}x{latent_width} is not divisible by patch {patch_size}.")

    text_token_tags = text_token_tags.to(device="cpu", dtype=torch.long).flatten()
    if not bool(torch.isin(text_token_tags, torch.tensor([VIDEO_TAG, TEXT_TAG])).all()):
        raise ValueError("MiniMax-H3 Ref2AV text_token_tags must be video=0 or text=1.")
    references = tuple(references)
    for index, reference in enumerate(references):
        if reference.kind not in {"image", "video", "audio"}:
            raise ValueError(f"references[{index}].kind must be image/video/audio, got {reference.kind!r}.")
        if reference.kind != "audio" and (
            reference.num_latent_frames <= 0 or reference.latent_height <= 0 or reference.latent_width <= 0 or reference.latent_height % patch_h or reference.latent_width % patch_w
        ):
            raise ValueError(f"references[{index}] has invalid visual geometry {reference}.")
        if reference.kind == "image" and reference.num_latent_frames != 1:
            raise ValueError(f"references[{index}] is an image and must have one latent frame.")
        if reference.kind == "audio" and reference.num_audio_latents <= 0:
            raise ValueError(f"references[{index}] is audio but has no audio latents.")

    num_text_tokens = int(text_token_tags.numel())
    target_frame_grid, target_width_grid = _frame_position_grid(
        latent_height,
        latent_width,
        patch_h,
        patch_w,
    )
    num_target_video_rows = num_latent_frames * target_frame_grid.shape[0]
    num_target_audio_rows = num_audio_latents * AUDIO_CHANNELS
    num_reference_video_rows = sum(reference.num_video_rows for reference in references)
    num_reference_audio_rows = sum(reference.num_audio_rows for reference in references)
    sequence_length = num_text_tokens + num_reference_video_rows + num_reference_audio_rows + num_target_audio_rows + num_target_video_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(
        num_text_tokens,
        dtype=torch.float64,
    )

    video_indices = []
    audio_indices = []
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for reference in references:
        if reference.kind == "image":
            rows = slice(cursor, cursor + reference.num_video_rows)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(
                reference.latent_height,
                reference.latent_width,
                patch_h,
                patch_w,
            )
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            rotary_time += 1.0
            continue

        if reference.kind == "audio":
            rows = slice(cursor, cursor + reference.num_audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(
                position_ids,
                rows,
                reference.num_audio_latents,
                rotary_time,
                target_width_grid,
            )
            rotary_time += float(reference.num_audio_latents)
            continue

        # A video's soundtrack and frames share one origin. Audio rows precede
        # video rows even though the aggregate modality tensors are separate.
        audio_rows = slice(cursor, cursor + reference.num_audio_rows)
        video_rows = slice(
            audio_rows.stop,
            audio_rows.stop + reference.num_video_rows,
        )
        cursor = video_rows.stop
        audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
        video_indices.append(torch.arange(video_rows.start, video_rows.stop))
        frame_grid, width_grid = _frame_position_grid(
            reference.latent_height,
            reference.latent_width,
            patch_h,
            patch_w,
        )
        _fill_audio_positions(
            position_ids,
            audio_rows,
            reference.num_audio_latents,
            rotary_time,
            width_grid,
        )
        frame_time = _temporal_grid(reference.num_latent_frames, rotary_time)
        position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
        position_ids[video_rows, 1:] = frame_grid.repeat(
            reference.num_latent_frames,
            1,
        )
        rotary_time += max(
            float(reference.num_audio_latents),
            _reference_temporal_position_span(reference.num_latent_frames),
        )

    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(
        position_ids,
        slice(audio_start, video_start),
        num_audio_latents,
        rotary_time,
        target_width_grid,
    )
    frame_time = _temporal_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = target_frame_grid.repeat(
        num_latent_frames,
        1,
    )

    video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG

    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_reference_video_rows,
        num_condition_audio_rows=num_reference_audio_rows,
    )


__all__ = [
    "MiniMaxH3ReferenceGeometry",
    "build_ref2av_packed_sequence",
]
