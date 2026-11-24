# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.sf3d.weights.modulation import torch


def Modulation_forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    emb = self.linear2(self.silu(self.linear1(condition)))
    scale, shift = torch.chunk(emb, 2, dim=1)
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x
