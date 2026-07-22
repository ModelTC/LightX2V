from dataclasses import dataclass

import torch


@dataclass
class ErnieImagePreInferOutput:
    hidden_states: torch.Tensor
    image_tokens_len: int
    image_hw: tuple[int, int]
    rotary_freqs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    rotary_positions: torch.Tensor | None
    temb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    conditioning: torch.Tensor
