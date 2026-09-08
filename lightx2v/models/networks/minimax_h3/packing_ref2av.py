"""Native MiniMax-H3 omni-reference preprocessing and packed geometry."""

import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from .packing import (
    AUDIO_CHANNELS,
    AUDIO_TAG,
    CANVAS_MULTIPLE,
    FPS,
    FRAMES_PER_CHUNK,
    LATENTS_PER_CHUNK,
    TEXT_TAG,
    VIDEO_TAG,
    MiniMaxH3PackedSequence,
    _spatial_position_grid,
    _temporal_position_grid,
    resolve_canvas_size,
)

REFERENCE_IMAGE_SHORT_EDGE = 2048
REFERENCE_IMAGE_RESIZE_MODES = ("match", "max", "diffusers")
DEFAULT_REFERENCE_IMAGE_RESIZE_MODE = "diffusers"
QWEN_VIDEO_SAMPLE_FPS = 2
QWEN_TEMPORAL_PATCH = 2
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_REFERENCES = 12
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)


@dataclass
class MiniMaxH3PreparedReference:
    kind: str
    has_audio: bool = False
    image: Any = None
    frames: Any = None
    waveform: torch.Tensor | None = None
    block_timestamps: list[float] = field(default_factory=list)
    num_latent_frames: int = 1
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0
    video_rows: torch.Tensor | None = None
    audio_rows: torch.Tensor | None = None

    @property
    def num_video_rows(self) -> int:
        return self.num_latent_frames * (self.latent_height // 2) * (self.latent_width // 2)

    @property
    def num_audio_rows(self) -> int:
        return self.num_audio_latents * AUDIO_CHANNELS


def _decode_reference_soundtrack(av, container, stream) -> tuple[torch.Tensor, int]:
    """Decode one container stream as planar float at its native rate."""

    sample_rate = int(stream.codec_context.sample_rate)
    resampler = av.audio.resampler.AudioResampler(
        format="fltp",
        layout=stream.layout,
        rate=sample_rate,
    )
    chunks = []
    for frame in container.decode(stream):
        chunks.extend(torch.from_numpy(value.to_ndarray()) for value in resampler.resample(frame))
    chunks.extend(torch.from_numpy(value.to_ndarray()) for value in resampler.resample(None))
    if not chunks:
        raise ValueError("The MiniMax-H3 reference audio stream contains no samples")
    return torch.cat(chunks, dim=-1).to(torch.float32), sample_rate


def decode_reference_video(media) -> tuple[np.ndarray, float, tuple[torch.Tensor, int] | None]:
    """Decode local reference video frames and its optional soundtrack."""

    path = os.fspath(media)
    if not os.path.isfile(path):
        raise ValueError(f"MiniMax-H3 reference video is not a local file: {path}")
    try:
        import av
    except ImportError as error:
        raise ImportError("Decoding a MiniMax-H3 reference video requires PyAV") from error

    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream to decode in {path}")
        stream = container.streams.video[0]
        frames, rotation = [], 0.0
        for frame in container.decode(stream):
            rotation = frame.rotation
            frames.append(frame.to_ndarray(format="rgb24"))
        frame_rate = float(stream.average_rate or stream.guessed_rate)
        soundtrack = None
        if container.streams.audio:
            container.seek(0)
            soundtrack = _decode_reference_soundtrack(av, container, container.streams.audio[0])

    if not frames:
        raise ValueError(f"No video frames to decode in {path}")
    frames = np.stack(frames)
    turns = round(rotation / 90.0) % 4
    if turns:
        frames = np.ascontiguousarray(np.rot90(frames, k=-turns, axes=(1, 2)))
    return frames, frame_rate, soundtrack


def decode_reference_audio(media) -> tuple[torch.Tensor, int]:
    """Decode a local audio reference at the sample rate it carries."""

    path = os.fspath(media)
    if not os.path.isfile(path):
        raise ValueError(f"MiniMax-H3 reference audio is not a local file: {path}")
    try:
        import av
    except ImportError as error:
        raise ImportError("Decoding a MiniMax-H3 reference audio file requires PyAV") from error

    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream to decode in {path}")
        return _decode_reference_soundtrack(av, container, container.streams.audio[0])


def _temporal_position_span(num_latent_frames: int) -> float:
    return sum(_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)] for index in range(num_latent_frames))


def _frame_position_grid(latent_height: int, latent_width: int, patch_h: int, patch_w: int):
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(position_ids, rows, num_audio_latents, rotary_time, width_grid):
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat(
        (
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        )
    )


def build_ref2av_packed_sequence(
    text_token_tags: torch.Tensor,
    references: list[MiniMaxH3PreparedReference],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> MiniMaxH3PackedSequence:
    """Build ``[presentation | ordered references | target audio | target video]``."""
    _, patch_h, patch_w = patch_size
    num_text_tokens = int(text_token_tags.shape[0])
    num_target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio_rows = num_audio_latents * AUDIO_CHANNELS
    num_reference_video_rows = sum(ref.num_video_rows for ref in references if ref.kind != "audio")
    num_reference_audio_rows = sum(ref.num_audio_rows for ref in references)
    sequence_length = num_text_tokens + num_reference_video_rows + num_reference_audio_rows + num_target_audio_rows + num_target_video_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    video_indices, audio_indices = [], []
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for reference in references:
        if reference.kind == "image":
            rows = slice(cursor, cursor + reference.num_video_rows)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(reference.latent_height, reference.latent_width, patch_h, patch_w)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            rotary_time += 1.0
        elif reference.kind == "audio":
            rows = slice(cursor, cursor + reference.num_audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(position_ids, rows, reference.num_audio_latents, rotary_time, target_width_grid)
            rotary_time += float(reference.num_audio_latents)
        elif reference.kind == "video":
            audio_rows = slice(cursor, cursor + reference.num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + reference.num_video_rows)
            cursor = video_rows.stop
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))
            frame_grid, width_grid = _frame_position_grid(reference.latent_height, reference.latent_width, patch_h, patch_w)
            _fill_audio_positions(position_ids, audio_rows, reference.num_audio_latents, rotary_time, width_grid)
            frame_time = _temporal_position_grid(reference.num_latent_frames, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[video_rows, 1:] = frame_grid.repeat(reference.num_latent_frames, 1)
            rotary_time += max(float(reference.num_audio_latents), _temporal_position_span(reference.num_latent_frames))
        else:
            raise ValueError(f"Unknown MiniMax-H3 reference kind: {reference.kind!r}")

    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(position_ids, slice(audio_start, video_start), num_audio_latents, rotary_time, target_width_grid)
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)
    video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.long()
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG
    return MiniMaxH3PackedSequence(
        sequence_length,
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_reference_video_rows,
        num_reference_audio_rows,
    )


def resolve_reference_image_size(
    width: int,
    height: int,
    *,
    target_width: int,
    target_height: int,
    mode: str = DEFAULT_REFERENCE_IMAGE_RESIZE_MODE,
    multiple: int = CANVAS_MULTIPLE,
    max_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
) -> tuple[int, int]:
    if width <= 0 or height <= 0 or width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be positive and within 1:4..4:1, got {width}x{height}")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"The target canvas must be positive, got {target_width}x{target_height}")
    if mode not in REFERENCE_IMAGE_RESIZE_MODES:
        raise ValueError(f"reference_image_resize_mode must be one of {REFERENCE_IMAGE_RESIZE_MODES}, got {mode!r}")

    if mode == "match":
        scale = min(1.0, math.sqrt((target_width * target_height) / (width * height)))
    elif mode == "max":
        scale = min(1.0, max_short_edge / min(width, height))
    else:
        scale = max_short_edge / min(width, height)
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def prepare_reference_image(image: Image.Image, height: int, width: int) -> Image.Image:
    return image if image.size == (width, height) else image.resize((width, height), Image.Resampling.LANCZOS)


def resample_reference_frames(frames: np.ndarray, fps: float) -> np.ndarray:
    if fps <= 0:
        raise ValueError(f"A reference video needs positive fps, got {fps}")
    if fps == FPS:
        return frames
    scale = FPS / fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    return np.repeat(frames, np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)), axis=0)


def prepare_reference_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Reference video must be [F,H,W,3], got {tuple(frames.shape)}")
    frames = frames[:num_frames]
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1])
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack([np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames])


def sample_reference_video_frames(frames: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    stride = FPS / QWEN_VIDEO_SAMPLE_FPS
    indices, cursor = [], 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    timestamps = [index / QWEN_VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % QWEN_TEMPORAL_PATCH)
    blocks = [(timestamps[index] + timestamps[index + QWEN_TEMPORAL_PATCH - 1]) / 2 for index in range(0, len(timestamps), QWEN_TEMPORAL_PATCH)]
    return [frames[index] for index in indices], blocks


def prepare_reference_waveform(waveform, sample_rate: int, target_sample_rate: int, max_duration: float):
    waveform = torch.as_tensor(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, AUDIO_CHANNELS):
        raise ValueError(f"Reference audio must be mono/stereo [C,S], got {tuple(waveform.shape)}")
    waveform = waveform.float()[:, : int(max_duration * sample_rate)]
    if waveform.shape[0] == 1:
        waveform = waveform.expand(AUDIO_CHANNELS, -1).contiguous()
    if sample_rate == target_sample_rate:
        return waveform
    try:
        import torchaudio
    except ImportError as error:
        raise ImportError("Resampling MiniMax-H3 reference audio requires torchaudio") from error
    return torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)


def trim_reference_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        raise ValueError("Reference video contains no frames")
    return max(1, (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK) * FRAMES_PER_CHUNK + LATENTS_PER_CHUNK


def build_ref2av_presentation(tokenizer, prompt, references, image_token_counts, video_block_token_counts):
    token_ids, token_tags = [], []

    def emit_text(value):
        ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        token_ids.extend(ids)
        token_tags.extend([TEXT_TAG] * len(ids))

    def emit_vision(pad_token, count):
        ids = [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
        ids += [tokenizer.convert_tokens_to_ids(pad_token)] * count
        ids += [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        token_ids.extend(ids)
        token_tags.extend([VIDEO_TAG] * len(ids))

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit_text(f"<Audio {counts['audio']}>: ")
        if reference.kind == "image":
            counts["image"] += 1
            emit_text(f"<Picture {counts['image']}>: ")
            emit_vision("<|image_pad|>", image_token_counts[counts["image"] - 1])
        elif reference.kind == "video":
            counts["video"] += 1
            emit_text(f"<Video {counts['video']}>: ")
            for timestamp in reference.block_timestamps:
                emit_text(f"<{timestamp:.1f} seconds>")
                emit_vision("<|video_pad|>", video_block_token_counts[counts["video"] - 1])
    emit_text(prompt)
    return token_ids, token_tags


__all__ = [name for name in globals() if name.startswith("MiniMaxH3") or name.startswith("MAX_")]
