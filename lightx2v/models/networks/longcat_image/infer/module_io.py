from dataclasses import dataclass

import torch


@dataclass
class LongCatImagePreInferModuleOutput:
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    temb: torch.Tensor
    image_rotary_emb: tuple  # (cos, sin) tuple
