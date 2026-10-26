"""Audio-only FL2VA row geometry used by TaoMate's Base10 teacher."""

import numpy as np
import torch

from lightx2v.models.networks.minimax_h3.packing import AUDIO_TAG, MiniMaxH3PackedSequence, _spatial_position_grid


def build_audio_only_layout(
    text_token_tags: torch.Tensor,
    audio_latents: int,
    latent_height: int,
    latent_width: int,
    *,
    previous_audio_latents: int | None = None,
    reference_latents: int = 40,
) -> MiniMaxH3PackedSequence:
    """Build the official valid rows in ``[text, clean reference, target]`` order.

    The official teacher pads the result to 64 rows and attends to that padding
    as a separate segment. LightX2V leaves the isolated padding segment out of
    the dense attention input; every valid row keeps its original position.
    """
    text_count = int(text_token_tags.numel())
    if text_count <= 0 or audio_latents <= 0:
        raise ValueError("Base10 needs nonempty text and target audio")
    if latent_height % 2 or latent_width % 2:
        raise ValueError("Base10 latent height and width must be even")
    if previous_audio_latents is not None and previous_audio_latents < reference_latents:
        raise ValueError("The previous audio request is shorter than the clean reference tail")

    reference_count = 0 if previous_audio_latents is None else reference_latents
    reference_rows = 2 * reference_count
    target_rows = 2 * audio_latents
    sequence_length = text_count + reference_rows + target_rows
    text_indices = torch.arange(text_count, dtype=torch.long)
    audio_indices = torch.arange(text_count, sequence_length, dtype=torch.long)
    position_ids = torch.zeros((sequence_length, 3), dtype=torch.float64)
    position_ids[text_indices, 0] = torch.arange(text_count, dtype=torch.float64)

    sqrt_area = np.sqrt(latent_height * latent_width)
    width_grid = _spatial_position_grid(latent_width, 2, sqrt_area)
    if reference_count:
        reference = slice(text_count, text_count + reference_rows)
        start = text_count + previous_audio_latents - reference_count
        position_ids[reference, 0] = (start + torch.arange(reference_count, dtype=torch.float64)).repeat(2)
        position_ids[reference.start : reference.start + reference_count, 2] = width_grid[0]
        position_ids[reference.start + reference_count : reference.stop, 2] = width_grid[-1]
    target = slice(text_count + reference_rows, sequence_length)
    start = text_count + (0 if previous_audio_latents is None else previous_audio_latents)
    position_ids[target, 0] = (start + torch.arange(audio_latents, dtype=torch.float64)).repeat(2)
    position_ids[target.start : target.start + audio_latents, 2] = width_grid[0]
    position_ids[target.start + audio_latents : target.stop, 2] = width_grid[-1]

    token_tags = torch.full((sequence_length,), AUDIO_TAG, dtype=torch.long)
    token_tags[:text_count] = text_token_tags.cpu().to(torch.long)
    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=torch.empty(0, dtype=torch.long),
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_audio_rows=reference_rows,
    )
