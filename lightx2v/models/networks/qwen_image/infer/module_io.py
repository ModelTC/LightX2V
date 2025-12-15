from dataclasses import dataclass

import torch


@dataclass
class QwenPreInferModuleOutput:
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    embed0: torch.Tensor
    image_rotary_emb: torch.Tensor
