from dataclasses import dataclass

import torch


@dataclass
class LingbotVAPreInferOutput:
    x: torch.Tensor
    context: torch.Tensor
    temb: torch.Tensor
    timestep_proj: torch.Tensor
    rotary_emb: torch.Tensor
    grid_shape: tuple[int, int, int]
    action_mode: bool
