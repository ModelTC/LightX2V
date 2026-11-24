# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
import torch
from torch import nn


class Modulation(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int, zero_init: bool = False, single_layer: bool = False):
        super().__init__()
        self.silu = nn.SiLU()
        if single_layer:
            self.linear1 = nn.Identity()
        else:
            self.linear1 = nn.Linear(condition_dim, condition_dim)
        self.linear2 = nn.Linear(condition_dim, embedding_dim * 2)
        if zero_init:
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        from lightx2v.models.networks.sf3d.infer.modulation import Modulation_forward

        return Modulation_forward(self, x, condition)


__all__ = ["torch"]
