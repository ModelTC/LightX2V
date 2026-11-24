# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.hymotion.weights.bricks import Tensor, torch


def RMSNorm__norm(self, x: Tensor) -> Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def RMSNorm_forward(self, x: Tensor) -> Tensor:
    output = self._norm(x.float()).type_as(x)
    if hasattr(self, "weight"):
        output = output * self.weight
    return output
