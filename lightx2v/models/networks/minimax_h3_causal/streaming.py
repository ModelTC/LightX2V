"""Chunk geometry and rolling temporal positions for causal H3."""

import math
from dataclasses import dataclass

import torch

from lightx2v.models.networks.minimax_h3.packing import AUDIO_TAG, VIDEO_TAG, MiniMaxH3PackedSequence


@dataclass(frozen=True)
class MiniMaxH3Chunk:
    video_frames: slice
    video_rows: slice
    audio_frames: slice
    positions: torch.Tensor

    @property
    def num_video_rows(self):
        return self.video_rows.stop - self.video_rows.start

    @property
    def num_audio_frames(self):
        return self.audio_frames.stop - self.audio_frames.start


class MiniMaxH3StreamingPlan:
    def __init__(self, layout, num_latent_frames, num_audio_latents, rows_per_frame, ar_config):
        self.sink_chunks = int(ar_config["sink_chunk_num"])
        self.latest_chunks = int(ar_config["latest_chunk_num"])
        self.max_rope_slot = int(ar_config["kv_cache_max_t_rope"])
        chunk_size = int(ar_config["num_frame_per_chunk"])
        if chunk_size != 5 or self.sink_chunks != 1 or self.latest_chunks != 2:
            raise ValueError("This H3 checkpoint requires 5 video latents per chunk, one sink chunk and two latest chunks")
        if self.max_rope_slot < self.latest_chunks - 1:
            raise ValueError("kv_cache_max_t_rope must fit the latest attention window")

        text_rows = layout.text_indices.numel()
        condition_rows = text_rows + layout.num_condition_video_rows
        self.condition_layout = MiniMaxH3PackedSequence(
            condition_rows,
            layout.position_ids[:condition_rows],
            layout.token_tags[:condition_rows],
            torch.arange(text_rows, condition_rows),
            torch.empty(0, dtype=torch.long),
            torch.arange(text_rows),
            layout.num_condition_video_rows,
        )
        video_positions = layout.position_ids[layout.video_indices[layout.num_condition_video_rows :]]
        audio_positions = layout.position_ids[layout.audio_indices]
        self.chunks = []
        audio_start = 0
        anchor = float(video_positions[0, 0])
        for start in range(0, num_latent_frames, chunk_size):
            end = min(start + chunk_size, num_latent_frames)
            if end == num_latent_frames:
                audio_end = num_audio_latents
            else:
                boundary = float(video_positions[end * rows_per_frame, 0]) - anchor
                rounded = round(boundary)
                audio_end = min(rounded if math.isclose(boundary, rounded, abs_tol=1e-9, rel_tol=0) else math.ceil(boundary), num_audio_latents)
            video_rows = slice(start * rows_per_frame, end * rows_per_frame)
            positions = torch.cat((video_positions[video_rows], audio_positions[audio_start:audio_end], audio_positions[num_audio_latents + audio_start : num_audio_latents + audio_end]))
            self.chunks.append(MiniMaxH3Chunk(slice(start, end), video_rows, slice(audio_start, audio_end), positions))
            audio_start = audio_end
        self.cache_size = max(self.prefix_positions(i).shape[0] + chunk.positions.shape[0] for i, chunk in enumerate(self.chunks))
        self.sink_rows = condition_rows + self.chunks[0].positions.shape[0]

    def positions(self, chunk_index, current_index):
        positions = self.chunks[chunk_index].positions
        if chunk_index < self.sink_chunks:
            return positions
        slot = min(current_index, self.max_rope_slot) - (current_index - chunk_index)
        shifted = positions.clone()
        shifted[:, 0] += self.chunks[slot].positions[0, 0] - positions[0, 0]
        return shifted

    def prefix_positions(self, index):
        kept = sorted(set(range(min(self.sink_chunks, index))) | set(range(max(0, index - self.latest_chunks + 1), index)))
        return torch.cat([self.condition_layout.position_ids] + [self.positions(i, index) for i in kept])

    def chunk_layout(self, index):
        chunk = self.chunks[index]
        positions = self.positions(index, index)
        video_rows = chunk.num_video_rows
        tags = torch.full((positions.shape[0],), AUDIO_TAG, dtype=torch.long)
        tags[:video_rows] = VIDEO_TAG
        return MiniMaxH3PackedSequence(
            positions.shape[0],
            positions,
            tags,
            torch.arange(video_rows),
            torch.arange(video_rows, positions.shape[0]),
            torch.empty(0, dtype=torch.long),
        )
