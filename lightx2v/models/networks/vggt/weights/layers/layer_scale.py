# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# See the Apache-2.0 LICENSE in the repository root.
# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110

from typing import Union

import torch
from torch import Tensor, nn


class LayerScaleWeights(nn.Module):
    _inference_type = ("lightx2v.models.networks.vggt.infer.layers.layer_scale", "LayerScaleInfer")

    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-05, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.vggt.infer.layers.layer_scale import LayerScaleInfer

        return LayerScaleInfer(self).infer(*args, **kwargs)
