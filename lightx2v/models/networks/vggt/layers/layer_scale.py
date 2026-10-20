# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# See the Apache-2.0 LICENSE in the repository root.
# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110

from typing import Union

import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-05, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
