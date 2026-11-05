from dataclasses import dataclass

import torch


@dataclass
class DreamZeroTimestepSegments:
    """Per-token time embeddings stored as unique rows plus a row->segment map."""

    seg_idx: torch.Tensor  # int32 [seq_total]
    embed: torch.Tensor  # [num_segments, dim]
    embed0: torch.Tensor  # [num_segments, 6, dim]


@dataclass
class DreamZeroPreInferOutput:
    x: torch.Tensor
    time_segments: DreamZeroTimestepSegments
    context: torch.Tensor | None
    freqs: torch.Tensor
    rope_positions: torch.Tensor | None
    freqs_action: torch.Tensor
    freqs_state: torch.Tensor
    grid_size: tuple[int, int, int]
    seq_len: int
    action_length: int
    action_register_length: int | None
    current_start_frame: int
    update_cache: bool
    cache_name: str
