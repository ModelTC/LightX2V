# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

import torch.nn as nn

from lightx2v.common.ops.sparse.basic import VarLenTensor
from lightx2v.common.ops.sparse.linear import SparseLinear
from lightx2v.common.ops.sparse.nonlinearity import SparseGELU


class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(SparseLinear(channels, int(channels * mlp_ratio)), SparseGELU(approximate="tanh"), SparseLinear(int(channels * mlp_ratio), channels))

    def forward(self, x: VarLenTensor) -> VarLenTensor:
        return self.mlp(x)
