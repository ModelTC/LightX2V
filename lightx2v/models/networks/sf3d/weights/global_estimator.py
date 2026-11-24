# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from dataclasses import dataclass, field
from typing import Any

from jaxtyping import Float
from torch import Tensor, nn

from lightx2v.models.networks.sf3d.utils.base import BaseModule
from lightx2v.models.networks.sf3d.weights.network import get_activation


@dataclass
class HeadSpec:
    name: str
    out_channels: int
    n_hidden_layers: int
    output_activation: str | None = None
    output_bias: float = 0.0
    add_to_decoder_features: bool = False
    shape: list[int] | None = None


class MultiHeadEstimator(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        triplane_features: int = 1024
        n_layers: int = 2
        hidden_features: int = 512
        activation: str = "relu"
        pool: str = "max"
        heads: list[HeadSpec] = field(default_factory=list)

    cfg: Config

    def configure(self):
        layers = []
        cur_features = self.cfg.triplane_features * 3
        for _ in range(self.cfg.n_layers):
            layers.append(nn.Conv2d(cur_features, self.cfg.hidden_features, kernel_size=3, padding=0, stride=2))
            layers.append(self.make_activation(self.cfg.activation))
            cur_features = self.cfg.hidden_features
        self.layers = nn.Sequential(*layers)
        assert len(self.cfg.heads) > 0
        heads = {}
        for head in self.cfg.heads:
            head_layers = []
            for i in range(head.n_hidden_layers):
                head_layers += [nn.Linear(self.cfg.hidden_features, self.cfg.hidden_features), self.make_activation(self.cfg.activation)]
            head_layers += [nn.Linear(self.cfg.hidden_features, head.out_channels)]
            heads[head.name] = nn.Sequential(*head_layers)
        self.heads = nn.ModuleDict(heads)

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError

    def forward(self, triplane: Float[Tensor, "B 3 F Ht Wt"]) -> dict[str, Any]:
        from lightx2v.models.networks.sf3d.infer.global_estimator import MultiHeadEstimator_forward

        return MultiHeadEstimator_forward(self, triplane)


__all__ = ["Any", "Float", "Tensor", "get_activation"]
