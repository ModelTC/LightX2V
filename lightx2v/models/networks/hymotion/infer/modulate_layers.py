# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.hymotion.weights.modulate_layers import Tensor


def ModulateDiT_forward(self, x: Tensor) -> Tensor:
    return self.linear(self.act(x))
