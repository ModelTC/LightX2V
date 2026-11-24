# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from dataclasses import dataclass, field
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torchvision.transforms import Normalize

from lightx2v.models.networks.sf3d.utils.base import BaseModule
from lightx2v.models.networks.sf3d.weights.network import get_activation

from ..infer.vision import openclip_image


@dataclass
class HeadSpec:
    name: str
    out_channels: int
    n_hidden_layers: int
    output_activation: str | None = None
    output_bias: float = 0.0
    add_to_decoder_features: bool = False
    shape: list[int] | None = None


class ClipBasedHeadEstimator(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        model: str = "ViT-B-32"
        pretrain: str = "laion2b_s34b_b79k"
        distribution: str = "beta"
        distribution_eval: str = "mode"
        activation: str = "relu"
        hidden_features: int = 512
        heads: list[HeadSpec] = field(default_factory=list)

    cfg: Config

    def configure(self):
        self.visual_weights = None
        assert len(self.cfg.heads) > 0
        heads = {}
        for head in self.cfg.heads:
            head_layers = []
            for i in range(head.n_hidden_layers):
                head_layers += [nn.Linear(self.cfg.hidden_features, self.cfg.hidden_features), self.make_activation(self.cfg.activation)]
            head_layers = [nn.Sequential(*head_layers)]
            head_layers += [
                nn.Sequential(nn.Linear(self.cfg.hidden_features, self.cfg.hidden_features), self.make_activation(self.cfg.activation), nn.Linear(self.cfg.hidden_features, 1)) for _ in range(2)
            ]
            heads[head.name] = nn.ModuleList(head_layers)
        self.heads = nn.ModuleDict(heads)

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError

    def forward(self, cond_image: Float[Tensor, "B 1 H W 3"], sample: bool = True) -> dict[str, Any]:
        from lightx2v.models.networks.sf3d.infer.image_estimator import ClipBasedHeadEstimator_forward

        return ClipBasedHeadEstimator_forward(self, cond_image, sample)


__all__ = ["Any", "Float", "Normalize", "Tensor", "get_activation", "nn", "openclip_image", "torch"]
