# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

import torch
from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MlpFP32(Mlp):
    @staticmethod
    def map_to_args_to_float(args, kwargs):
        args = tuple(torch.float32 if isinstance(arg, torch.dtype) else arg for arg in args)
        kwargs = dict(kwargs)
        for key in kwargs:
            if key == "dtype":
                kwargs[key] = torch.float32
        return args, kwargs

    def to(self, *args, **kwargs):
        self.fc1 = self.fc1.to(*args, **kwargs)
        args, kwargs = self.map_to_args_to_float(args, kwargs)
        self.fc2 = self.fc2.to(*args, **kwargs)
        return self

    def forward_infer(self, x):
        x = self.fc1(x)
        x = 0.5 * x * (1 + torch.erf(x * 2**-0.5))
        x = self.fc2(x.float())
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_infer(x)
