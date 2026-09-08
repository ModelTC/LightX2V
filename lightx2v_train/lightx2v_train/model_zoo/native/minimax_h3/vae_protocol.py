"""Shared MiniMax-H3 latent normalization, tiled decoding, and checkpoint protocol."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .vae_geometry import (
    FRAME_OVERLAP,
    FRAME_PRE_PADDING,
    TEMPORAL_COMPRESSION_RATIO,
    TOKEN_DROP,
    TOKEN_OVERLAP,
    TOKENS_CHUNK_SIZE,
    spatial_tile_layout,
    validate_aligned_num_frames,
    video_num_frames_from_latent,
)


class MiniMaxH3VideoVAE(nn.Module):
    """H3 video interface; subclasses supply the decoder and latent statistics."""

    config_name = "config.json"
    weights_name = "model.safetensors"

    def initialize_post_quant_conv(self, teacher_vae) -> None:
        self.post_quant_conv.load_state_dict(teacher_vae.post_quant_conv.state_dict())

    def enable_gradient_checkpointing(self) -> None:
        self.decoder.enable_gradient_checkpointing()

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.float()
        mean = self.latents_mean.to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        std = self.latents_std.to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        return latents * std + mean

    def postprocess(self, video: torch.Tensor) -> torch.Tensor:
        video = video.float()
        mean = self.pixel_mean.to(device=video.device, dtype=video.dtype).view(1, 3, 1, 1, 1)
        std = self.pixel_std.to(device=video.device, dtype=video.dtype).view(1, 3, 1, 1, 1)
        return (video * std + mean).clamp(0.0, 1.0)

    def forward(self, normalized_latents: torch.Tensor, *, return_features: bool = False):
        vae_latents = self.denormalize_latents(normalized_latents)
        return self.decoder(self.post_quant_conv(vae_latents), return_features=return_features)

    @staticmethod
    def blend(left: torch.Tensor, right: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
        extent = min(left.shape[dim], right.shape[dim], extent)
        if extent == 0:
            return right
        positions = torch.arange(extent, device=right.device, dtype=right.dtype)
        shape = [1] * right.ndim
        shape[dim] = extent
        left_weight = (1.0 - positions / extent).view(shape)
        right_weight = (positions / extent).view(shape)
        left_slice = [slice(None)] * left.ndim
        right_slice = [slice(None)] * right.ndim
        left_slice[dim] = slice(-extent, None)
        right_slice[dim] = slice(0, extent)
        blended = left[tuple(left_slice)] * left_weight + right[tuple(right_slice)] * right_weight
        if extent == right.shape[dim]:
            return blended
        right_slice[dim] = slice(extent, None)
        return torch.cat((blended, right[tuple(right_slice)]), dim=dim)

    def stitch_tiles(
        self,
        rows: list[list[torch.Tensor]],
        height_overlaps: tuple[int, ...],
        width_overlaps: tuple[int, ...],
    ) -> torch.Tensor:
        output_rows = []
        for row_index, row in enumerate(rows):
            output_row = []
            for column_index, tile in enumerate(row):
                if row_index:
                    tile = self.blend(rows[row_index - 1][column_index], tile, height_overlaps[row_index - 1], -2)
                if column_index:
                    tile = self.blend(row[column_index - 1], tile, width_overlaps[column_index - 1], -1)
                if row_index < len(rows) - 1:
                    tile = tile[..., : -height_overlaps[row_index], :]
                if column_index < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[column_index]]
                output_row.append(tile)
            output_rows.append(torch.cat(output_row, dim=-1))
        return torch.cat(output_rows, dim=-2)

    def _decode_spatial_clip(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self(latents)

        layout = spatial_tile_layout(
            latents.shape[-2],
            latents.shape[-1],
            tile_height=self.tile_sample_min_height,
            tile_width=self.tile_sample_min_width,
            overlap_height=self.tile_sample_min_overlap_height,
            overlap_width=self.tile_sample_min_overlap_width,
        )
        rows = []
        for height_slice in layout.height_slices:
            row = []
            for width_slice in layout.width_slices:
                row.append(self(latents[..., height_slice, width_slice]))
            rows.append(row)
        return self.stitch_tiles(rows, layout.height_overlaps, layout.width_overlaps)

    def decode_raw(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        if normalized_latents.ndim != 5 or normalized_latents.shape[1] != self.post_quant_conv.in_channels:
            raise ValueError(
                f"MiniMax-H3 video latents must be [B,{self.post_quant_conv.in_channels},T,H,W], "
                f"got {tuple(normalized_latents.shape)}."
            )

        expected_frames = video_num_frames_from_latent(normalized_latents.shape[2])
        validate_aligned_num_frames(expected_frames)
        num_tokens = normalized_latents.shape[2] + TOKEN_DROP
        pad_tokens = (-num_tokens) % TOKENS_CHUNK_SIZE
        num_clips = (num_tokens + pad_tokens) // TOKENS_CHUNK_SIZE - int(TOKEN_DROP > 0)
        if pad_tokens:
            normalized_latents = torch.cat(
                (
                    normalized_latents,
                    normalized_latents[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1),
                ),
                dim=2,
            )

        chunk_num_frames = TOKENS_CHUNK_SIZE * TEMPORAL_COMPRESSION_RATIO
        decoded_chunks = []
        overlap = None
        for clip_index in range(num_clips):
            start = clip_index * TOKENS_CHUNK_SIZE
            clip = self._decode_spatial_clip(
                normalized_latents[:, :, start : start + TOKENS_CHUNK_SIZE + TOKEN_OVERLAP]
            )
            for overlap_index in range(int(TOKEN_DROP > 0) + 1):
                frame_start = overlap_index * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, FRAME_PRE_PADDING:]
                if overlap_index == 0:
                    if overlap is not None:
                        chunk = self.blend(overlap, chunk, FRAME_OVERLAP, -3)
                    decoded_chunks.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded_chunks.append(overlap)

        decoded = torch.cat(decoded_chunks, dim=2)
        if pad_tokens:
            intra_tail = 17 % TEMPORAL_COMPRESSION_RATIO
            num_tokens_before_pad = normalized_latents.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail if (num_tokens_before_pad + offset) % TOKENS_CHUNK_SIZE == 0 else TEMPORAL_COMPRESSION_RATIO
                for offset in range(pad_tokens)
            )
            decoded = decoded[:, :, :-pad_frames]
        if decoded.shape[2] != expected_frames:
            raise RuntimeError(f"H3 decoder produced {decoded.shape[2]} frames, expected {expected_frames}.")
        return decoded

    def decode(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        return self.postprocess(self.decode_raw(normalized_latents))

    def save_config(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        with (save_directory / self.config_name).open("w", encoding="utf-8") as handle:
            json.dump({**self.architecture_config, "use_tiling": self.use_tiling}, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def save_pretrained(self, save_directory: str | Path, *, state_dict=None, safe_serialization: bool = True) -> None:
        if not safe_serialization:
            raise ValueError("MiniMax-H3 VAE checkpoints use safetensors.")
        save_directory = Path(save_directory)
        self.save_config(save_directory)
        weights = self.state_dict() if state_dict is None else state_dict
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in weights.items()},
            str(save_directory / self.weights_name),
            metadata={"format": "pt"},
        )

    @classmethod
    def from_pretrained(cls, model_path: str | Path, *, device=None, dtype=None):
        model_path = Path(model_path)
        with (model_path / cls.config_name).open("r", encoding="utf-8") as handle:
            model = cls(**json.load(handle))
        model.load_state_dict(load_file(str(model_path / cls.weights_name), device="cpu"), strict=True)
        return model.to(device=device, dtype=dtype)
