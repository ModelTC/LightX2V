from dataclasses import dataclass

import torch


@dataclass
class Hunyuan3DPreInferOutput:
    hidden_states: torch.Tensor
    cond: torch.Tensor
    c: torch.Tensor
