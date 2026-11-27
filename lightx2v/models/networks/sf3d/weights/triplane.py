# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
import math
from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from lightx2v.models.networks.sf3d.utils.base import BaseModule


class TriplaneLearnablePositionalEmbedding(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        plane_size: int = 96
        num_channels: int = 1024

    cfg: Config

    def configure(self) -> None:
        self.embeddings = nn.Parameter(torch.randn((3, self.cfg.num_channels, self.cfg.plane_size, self.cfg.plane_size), dtype=torch.float32) * 1 / math.sqrt(self.cfg.num_channels))

    def forward(self, batch_size: int) -> Float[Tensor, "B Ct Nt"]:
        from lightx2v.models.networks.sf3d.infer.triplane import TriplaneLearnablePositionalEmbedding_forward

        return TriplaneLearnablePositionalEmbedding_forward(self, batch_size)

    def detokenize(self, tokens: Float[Tensor, "B Ct Nt"]) -> Float[Tensor, "B 3 Ct Hp Wp"]:
        from lightx2v.models.networks.sf3d.infer.triplane import TriplaneLearnablePositionalEmbedding_detokenize

        return TriplaneLearnablePositionalEmbedding_detokenize(self, tokens)


__all__ = ["Float", "Tensor", "rearrange", "repeat"]
