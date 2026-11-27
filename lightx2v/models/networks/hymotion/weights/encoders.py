# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.

import torch
from torch import Tensor, nn

from lightx2v.models.networks.worldmirror.models.layers.mlp import Mlp

from .bricks import get_activation_layer
from .modulate_layers import ModulateDiT, modulate


class MLP(Mlp):
    """Map released checkpoint dimensions to the existing two-layer MLP."""

    def __init__(self, in_dim: int, feat_dim: int, out_dim: int | None = None, act_type: str = "gelu", bias: bool = True, drop: float = 0.0):
        super().__init__(in_features=in_dim, hidden_features=feat_dim, out_features=out_dim, act_layer=get_activation_layer(act_type), bias=bias, drop=drop)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int, num_layers: int, act_type: str = "silu") -> None:
        super().__init__()
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        linears = []
        linears.append(nn.Linear(in_features=in_dim, out_features=self.feat_dim))
        for i in range(num_layers - 1):
            linears.append(get_activation_layer(act_type)())
            linears.append(nn.Linear(self.feat_dim, self.feat_dim))
        self.linears = nn.Sequential(*linears)

    def forward(self, x: Tensor) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.encoders import MLPEncoder_forward

        return MLPEncoder_forward(self, x)


class FinalLayer(nn.Module):
    def __init__(self, feat_dim: int, out_dim: int, act_type: str = "gelu", zero_init=False, **kwargs):
        super().__init__()
        self.norm_final = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-06)
        self.adaLN_modulation = ModulateDiT(feat_dim, factor=2, act_type=act_type)
        self.linear = nn.Linear(feat_dim, out_dim, bias=True)
        if zero_init:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, adapter: Tensor) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.encoders import FinalLayer_forward

        return FinalLayer_forward(self, x, adapter)


class TimestepEmbeddingEncoder(nn.Module):
    def __init__(self, embedding_dim: int, feat_dim: int, act_type: str = "silu", time_factor: float = 1.0) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.feat_dim = feat_dim
        self.time_factor = time_factor
        blocks = [nn.Linear(embedding_dim, self.feat_dim), get_activation_layer(act_type)(), nn.Linear(self.feat_dim, self.feat_dim)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, t: Tensor) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.encoders import TimestepEmbeddingEncoder_forward

        return TimestepEmbeddingEncoder_forward(self, t)

    @staticmethod
    def sinusodial_embedding(timesteps: Tensor, embedding_dim: int, temperature: float = 10000.0, time_factor: float = 1.0) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.encoders import TimestepEmbeddingEncoder_sinusodial_embedding

        return TimestepEmbeddingEncoder_sinusodial_embedding(timesteps, embedding_dim, temperature, time_factor)


__all__ = ["Tensor", "modulate", "torch"]
