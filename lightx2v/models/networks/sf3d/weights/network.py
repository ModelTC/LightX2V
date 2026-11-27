# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, Optional  # noqa: UP035 - exported aliases used by the split infer modules

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torch.amp import custom_fwd
from torch.autograd import Function

from lightx2v.models.networks.sf3d.utils.base import BaseModule, normalize
from lightx2v.models.networks.sf3d.utils.image import get_device


def conditional_decorator(decorator_with_args, condition, *args, **kwargs):
    def wrapper(fn):
        if condition:
            if len(kwargs) == 0:
                return decorator_with_args
            return decorator_with_args(*args, **kwargs)(fn)
        else:
            return fn

    return wrapper


class PixelShuffleUpsampleNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 1024
        out_channels: int = 40
        scale_factor: int = 4
        conv_layers: int = 4
        conv_kernel_size: int = 3

    cfg: Config

    def configure(self) -> None:
        layers = []
        output_channels = self.cfg.out_channels * self.cfg.scale_factor**2
        in_channels = self.cfg.in_channels
        for i in range(self.cfg.conv_layers):
            cur_out_channels = in_channels if i != self.cfg.conv_layers - 1 else output_channels
            layers.append(nn.Conv2d(in_channels, cur_out_channels, self.cfg.conv_kernel_size, padding=(self.cfg.conv_kernel_size - 1) // 2))
            if i != self.cfg.conv_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        layers.append(nn.PixelShuffle(self.cfg.scale_factor))
        self.upsample = nn.Sequential(*layers)

    def forward(self, triplanes: Float[Tensor, "B 3 Ci Hp Wp"]) -> Float[Tensor, "B 3 Co Hp2 Wp2"]:
        from lightx2v.models.networks.sf3d.infer.network import PixelShuffleUpsampleNetwork_forward

        return PixelShuffleUpsampleNetwork_forward(self, triplanes)


class _TruncExp(Function):
    @staticmethod
    @conditional_decorator(custom_fwd, get_device() == "cuda", cast_inputs=torch.float32, device_type=get_device())
    def forward(ctx, x):
        from lightx2v.models.networks.sf3d.infer.network import _TruncExp_forward

        return _TruncExp_forward(ctx, x)


trunc_exp = _TruncExp.apply


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none" or name == "linear" or name == "identity":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(x > 0.0031308, torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * x).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    elif name == "negative":
        return lambda x: -x
    elif name == "normalize_channel_last":
        return lambda x: normalize(x)
    elif name == "normalize_channel_first":
        return lambda x: normalize(x, dim=1)
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


@dataclass
class HeadSpec:
    name: str
    out_channels: int
    n_hidden_layers: int
    output_activation: str | None = None
    out_bias: float = 0.0


class MaterialMLP(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 120
        n_neurons: int = 64
        activation: str = "silu"
        heads: list[HeadSpec] = field(default_factory=list)

    cfg: Config

    def configure(self) -> None:
        assert len(self.cfg.heads) > 0
        heads = {}
        for head in self.cfg.heads:
            head_layers = []
            for i in range(head.n_hidden_layers):
                head_layers += [nn.Linear(self.cfg.in_channels if i == 0 else self.cfg.n_neurons, self.cfg.n_neurons), self.make_activation(self.cfg.activation)]
            head_layers += [nn.Linear(self.cfg.n_neurons, head.out_channels)]
            heads[head.name] = nn.Sequential(*head_layers)
        self.heads = nn.ModuleDict(heads)

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError

    def keys(self):
        return self.heads.keys()

    def forward(self, x, include: list | None = None, exclude: list | None = None):
        from lightx2v.models.networks.sf3d.infer.network import MaterialMLP_forward

        return MaterialMLP_forward(self, x, include, exclude)


__all__ = ["Float", "List", "Optional", "Tensor", "conditional_decorator", "custom_fwd", "get_activation", "get_device", "rearrange", "torch"]
