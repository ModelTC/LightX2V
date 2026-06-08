from dataclasses import dataclass

import torch


@dataclass
class ErnieImagePreInferOutput:
    hidden_states: torch.Tensor
    image_tokens_len: int
    image_hw: tuple[int, int]
    rotary_pos_emb: torch.Tensor
    temb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    conditioning: torch.Tensor
