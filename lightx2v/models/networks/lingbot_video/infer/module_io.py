from dataclasses import dataclass

import torch


@dataclass
class LingBotVideoPreInferOutput:
    hidden_states: torch.Tensor
    rotary_emb: torch.Tensor
    temb6: torch.Tensor
    temb_input: torch.Tensor
    n_video: int
    grid_t: int
    grid_h: int
    grid_w: int
    latent_shape: tuple
    temb_input: torch.Tensor
    temb6: torch.Tensor
    rotary_emb: torch.Tensor
    n_video: int
    grid_t: int
    grid_h: int
    grid_w: int
    latent_shape: tuple
