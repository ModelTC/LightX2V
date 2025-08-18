from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DefaultLinear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


def replace_linear_with_custom(model: nn.Module, CustomLinear: Type[nn.Module]) -> nn.Module:
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None

            custom_linear = CustomLinear(in_features=in_features, out_features=out_features, bias=bias)

            with torch.no_grad():
                custom_linear.weight.copy_(module.weight)
                if bias:
                    custom_linear.bias.copy_(module.bias)

            setattr(model, name, custom_linear)
        else:
            replace_linear_with_custom(module, CustomLinear)

    return model
