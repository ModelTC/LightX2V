# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from functools import partial

import torch
from torch import Tensor, nn

from ..utils.misc import to_2tuple
from .bricks import get_activation_layer, get_norm_layer
from .modulate_layers import ModulateDiT, modulate


class MLP(nn.Module):
    def __init__(
        self, in_dim: int, feat_dim: int, out_dim: int | None = None, act_type: str = "gelu", norm_type: str | None = None, bias: bool = True, drop: float = 0.0, use_conv: bool = False
    ) -> None:
        super().__init__()
        out_dim = out_dim or in_dim
        feat_dim = feat_dim or in_dim
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.fc1 = linear_layer(in_dim, feat_dim, bias=bias[0] if isinstance(bias, (list, tuple)) else bias)
        self.act = get_activation_layer(act_type)()
        self.drop1 = nn.Dropout(drop_probs[0] if isinstance(drop_probs, (list, tuple)) else drop_probs)
        self.norm = get_norm_layer(norm_type)(feat_dim) if norm_type else nn.Identity()
        self.fc2 = linear_layer(feat_dim, out_dim, bias=bias[1] if isinstance(bias, (list, tuple)) else bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.encoders import MLP_forward

        return MLP_forward(self, x)


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
