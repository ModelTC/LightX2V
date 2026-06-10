from dataclasses import dataclass

import torch


@dataclass
class DreamZeroPreInferOutput:
    x: torch.Tensor
    embed: torch.Tensor
    embed0: torch.Tensor
    context: torch.Tensor
    freqs: torch.Tensor
    freqs_action: torch.Tensor
    freqs_state: torch.Tensor
    grid_size: tuple[int, int, int]
    seq_len: int
    action_length: int
    action_register_length: int | None
    current_start_frame: int
    update_cache: bool
    cache_name: str
