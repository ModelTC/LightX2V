# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from dataclasses import dataclass, field

import torch
from torch import nn

from lightx2v.models.networks.sf3d.utils.base import BaseModule


class LinearCameraEmbedder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 25
        out_channels: int = 768
        conditions: list[str] = field(default_factory=list)

    cfg: Config

    def configure(self) -> None:
        self.linear = nn.Linear(self.cfg.in_channels, self.cfg.out_channels)

    def forward(self, **kwargs):
        from lightx2v.models.networks.sf3d.infer.camera import LinearCameraEmbedder_forward

        return LinearCameraEmbedder_forward(self, **kwargs)


__all__ = ["torch"]
