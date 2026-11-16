"""Padless text/reference/target packing for MiniMax-H3-Image."""

import numpy as np
import torch

from lightx2v.models.networks.minimax_h3.packing import (
    VIDEO_TAG,
    MiniMaxH3PackedSequence,
    _spatial_position_grid,
    patchify_video_latents,
    unpatchify_video_tokens,
)


def build_image_packed_sequence(text_token_tags, reference_grids, latent_height, latent_width, patch_size=(1, 2, 2)):
    """Return ``[Qwen rows | reference images | target image]`` metadata."""
    if tuple(patch_size) != (1, 2, 2):
        raise ValueError("MiniMax-H3-Image requires patch_size=(1, 2, 2)")
    grids = [*reference_grids, (latent_height, latent_width)]
    num_text = text_token_tags.numel()
    row_counts = [(h // 2) * (w // 2) for h, w in grids]
    num_reference_rows = sum(row_counts[:-1])
    sequence_length = num_text + sum(row_counts)
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text, 0] = torch.arange(num_text, dtype=torch.float64)
    cursor = num_text
    for image_index, ((height, width), rows) in enumerate(zip(grids, row_counts)):
        sqrt_area = np.sqrt(height * width)
        axes = [_spatial_position_grid(side, 2, sqrt_area) for side in (height, width)]
        grid = torch.stack([axis.reshape(-1) for axis in torch.meshgrid(*axes, indexing="ij")], dim=-1)
        position_ids[cursor : cursor + rows, 0] = float(num_text + image_index)
        position_ids[cursor : cursor + rows, 1:] = grid
        cursor += rows
    text_indices = torch.arange(num_text, dtype=torch.long)
    image_indices = torch.arange(num_text, sequence_length, dtype=torch.long)
    token_tags = torch.full((sequence_length,), VIDEO_TAG, dtype=torch.long)
    token_tags[:num_text] = text_token_tags.to(device="cpu", dtype=torch.long)
    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=image_indices,
        audio_indices=torch.empty(0, dtype=torch.long),
        text_indices=text_indices,
        num_condition_video_rows=num_reference_rows,
    )


__all__ = ["build_image_packed_sequence", "patchify_video_latents", "unpatchify_video_tokens"]
