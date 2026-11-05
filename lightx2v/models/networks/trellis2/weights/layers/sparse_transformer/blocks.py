# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

import torch.nn as nn

from lightx2v.common.ops.sparse.linear import SparseLinear
from lightx2v.common.ops.sparse.nonlinearity import SparseGELU


class SparseFeedForwardNetWeights(nn.Module):
    _inference_type = ("lightx2v.models.networks.trellis2.infer.layers.sparse_transformer.blocks", "SparseFeedForwardNetInfer")

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(SparseLinear(channels, int(channels * mlp_ratio)), SparseGELU(approximate="tanh"), SparseLinear(int(channels * mlp_ratio), channels))

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.trellis2.infer.layers.sparse_transformer.blocks import SparseFeedForwardNetInfer

        return SparseFeedForwardNetInfer(self).infer(*args, **kwargs)
