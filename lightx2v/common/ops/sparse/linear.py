# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

import torch.nn as nn

from lightx2v.common.ops.inference_module import InferenceLinear
from lightx2v.common.ops.sparse import VarLenTensor

__all__ = ["SparseLinear"]


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))


class InferenceSparseLinear(InferenceLinear):
    def forward(self, x):
        return x.replace(super().forward(x.feats))


def bind_sparse_linear_operators(module, backend="Default"):
    for name, child in list(module.named_modules()):
        if isinstance(child, SparseLinear):
            module.set_submodule(name, InferenceSparseLinear(child, name, backend))
