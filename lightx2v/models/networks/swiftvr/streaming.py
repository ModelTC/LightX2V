"""Causal chunking and streaming state for SwiftVR restoration."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed as dist

from .parallel import exchange_chunk_boundary
from .reae import StreamingAutoencoder


class ChunkType(Enum):
    FIRST = "first"
    MIDDLE = "middle"
    LAST = "last"


@dataclass(frozen=True)
class VideoChunk:
    type: ChunkType
    start: int
    frame_count: int
    index: int

    @property
    def is_first(self) -> bool:
        return self.index == 0

    @property
    def is_last(self) -> bool:
        return self.type is ChunkType.LAST

    @property
    def latent_count(self) -> int:
        return (self.frame_count - 1) // 4 + 1

    def output_range(self, raw_frame_count: int, frames_to_trim: int) -> range:
        start = max(0, self.start - frames_to_trim)
        stop = min(raw_frame_count, self.start + self.frame_count - (0 if self.is_last else frames_to_trim))
        return range(start, stop)


def padded_frame_count(frame_count: int) -> int:
    """Return the smallest ``4k+1`` count that contains all input frames."""

    return ((frame_count - 1 + 3) // 4) * 4 + 1


def build_video_chunks(frame_count: int, clip_length: int) -> list[VideoChunk]:
    if clip_length % 4:
        raise ValueError(f"SwiftVR clip_len must be a multiple of 4, got {clip_length}")
    if frame_count <= clip_length + 4:
        return [VideoChunk(ChunkType.LAST, 0, frame_count, 0)]

    chunks = [VideoChunk(ChunkType.FIRST, 0, clip_length + 4, 0)]
    start = clip_length + 4
    while start < frame_count:
        remaining = frame_count - start
        chunk_type = ChunkType.LAST if remaining <= clip_length else ChunkType.MIDDLE
        chunk_frames = remaining if chunk_type is ChunkType.LAST else clip_length
        chunks.append(VideoChunk(chunk_type, start, chunk_frames, len(chunks)))
        start += chunk_frames
    return chunks


class StreamingTransformer:
    def __init__(self, model, condition, overlap: int = 0):
        self.model = model
        self.condition = condition
        self.overlap = overlap
        self.reset()

    def reset(self):
        self.previous_latents = None

    @torch.inference_mode()
    def prepare_input(self, latents: torch.Tensor, chunk: VideoChunk, clip_latents: int, chunk_p_group=None) -> tuple[torch.Tensor, int]:
        """Advance input history before prediction so consecutive chunks can run concurrently."""
        low_quality = latents.permute(0, 2, 1, 3, 4).contiguous()
        rank = dist.get_rank(chunk_p_group) if chunk_p_group is not None else 0
        history = low_quality
        if chunk_p_group is not None:
            # The first chunk has one extra latent; pad other boundaries to the same shape.
            boundary = low_quality.new_empty((*low_quality.shape[:2], clip_latents + 1, *low_quality.shape[3:]))
            boundary[:, :, : low_quality.shape[2]].copy_(low_quality)
            boundary[:, :, low_quality.shape[2] :].zero_()
            boundary = exchange_chunk_boundary(boundary, chunk_p_group)
            if rank != 0:
                previous_count = clip_latents + (chunk.index == 1)
                self.previous_latents = boundary[:, :, :previous_count]
            history = boundary[:, :, :clip_latents]
        temporal_offset = chunk.start // 4
        if chunk.is_last:
            padding = clip_latents + 1 - chunk.latent_count
            if padding:
                if self.previous_latents is None:
                    prefix = low_quality.new_zeros((*low_quality.shape[:2], padding, *low_quality.shape[3:]))
                else:
                    prefix = self.previous_latents[:, :, -padding:]
                low_quality = torch.cat([prefix, low_quality], dim=2)
            return low_quality, max(0, temporal_offset - padding)

        previous_input = self.previous_latents[:, :, -self.overlap :] if self.previous_latents is not None and self.overlap else None
        overlap = previous_input.shape[2] if previous_input is not None else 0
        model_input = torch.cat([previous_input, low_quality], dim=2) if overlap else low_quality
        if rank == 0:
            # Only rank 0 carries history across batches; other ranks receive it from their predecessor.
            history = history if self.overlap else history[:, :, -clip_latents:]
            self.previous_latents = history.detach().clone()
        return model_input, temporal_offset - overlap


class SwiftVRRestorer:
    def __init__(
        self,
        autoencoder,
        model,
        prompt_embedding,
        overlap: int = 0,
        reae_frame_batch_size: int = 1,
    ):
        self.autoencoder = StreamingAutoencoder(autoencoder, reae_frame_batch_size)
        self.transformer = StreamingTransformer(model, model.prepare_condition(prompt_embedding), overlap)

    def reset(self):
        self.autoencoder.reset()
        self.transformer.reset()

    @torch.inference_mode()
    def encode_chunk(self, video: torch.Tensor, chunk: VideoChunk, clip_latents: int) -> tuple[torch.Tensor, int]:
        latents = self.autoencoder.encode(video, chunk.is_last)
        return self.transformer.prepare_input(latents, chunk, clip_latents, self.autoencoder.chunk_p_group)

    @torch.inference_mode()
    def decode_chunk(self, latents: torch.Tensor, chunk: VideoChunk, output_batch_size: int = 0) -> Iterator[torch.Tensor]:
        latents = latents[:, :, -chunk.latent_count :].permute(0, 2, 1, 3, 4).contiguous()
        return self.autoencoder.decode(latents, chunk.is_first, output_batch_size)

    @torch.inference_mode()
    def restore_chunk(self, video: torch.Tensor, chunk: VideoChunk, clip_latents: int) -> torch.Tensor:
        model_input, temporal_offset = self.encode_chunk(video, chunk, clip_latents)
        prediction = self.transformer.model.predict(model_input, self.transformer.condition, temporal_offset)
        return next(self.decode_chunk(model_input - prediction, chunk))
