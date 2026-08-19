"""Causal chunking and streaming state for SwiftVR restoration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

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
        self.temporal_offset = 0
        self.previous_input = None
        self.previous_latents = None

    @torch.inference_mode()
    def restore(self, latents: torch.Tensor, clip_latents: int) -> torch.Tensor:
        low_quality = latents.permute(0, 2, 1, 3, 4).contiguous()
        overlap = self.previous_input.shape[2] if self.previous_input is not None and self.overlap else 0
        model_input = torch.cat([self.previous_input.to(low_quality.device), low_quality], dim=2) if overlap else low_quality

        restored = model_input - self.model.predict(model_input, self.condition, self.temporal_offset - overlap)
        if overlap:
            restored = restored[:, :, overlap:]

        keep = min(self.overlap, low_quality.shape[2])
        self.previous_input = low_quality[:, :, -keep:].detach().cpu().clone() if keep else None
        self.previous_latents = low_quality[:, :, -clip_latents:].detach().cpu().clone()
        self.temporal_offset += low_quality.shape[2]
        return restored.permute(0, 2, 1, 3, 4).contiguous()

    @torch.inference_mode()
    def restore_last(self, latents: torch.Tensor, latent_count: int, clip_latents: int) -> torch.Tensor:
        low_quality = latents.permute(0, 2, 1, 3, 4).contiguous()
        padding = clip_latents + 1 - latent_count
        if padding:
            if self.previous_latents is None:
                prefix = torch.zeros(
                    low_quality.shape[0],
                    low_quality.shape[1],
                    padding,
                    low_quality.shape[3],
                    low_quality.shape[4],
                    dtype=low_quality.dtype,
                    device=low_quality.device,
                )
            else:
                prefix = self.previous_latents[:, :, -padding:].to(low_quality.device)
            low_quality = torch.cat([prefix, low_quality], dim=2)

        restored = low_quality - self.model.predict(
            low_quality,
            self.condition,
            max(0, self.temporal_offset - padding),
        )
        self.temporal_offset += latent_count
        return restored[:, :, -latent_count:].permute(0, 2, 1, 3, 4).contiguous()


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
    def restore_chunk(self, video: torch.Tensor, chunk: VideoChunk, clip_latents: int) -> torch.Tensor:
        latents = self.autoencoder.encode(video, chunk.is_last)
        if chunk.is_last:
            latents = self.transformer.restore_last(latents, chunk.latent_count, clip_latents)
        else:
            latents = self.transformer.restore(latents, clip_latents)
        return self.autoencoder.decode(latents, chunk.is_first)
