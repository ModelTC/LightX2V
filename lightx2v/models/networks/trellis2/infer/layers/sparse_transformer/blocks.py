# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

from lightx2v.common.ops.sparse.basic import VarLenTensor
from lightx2v.common.transformer_infer.module_infer import ModuleInfer


class SparseFeedForwardNetInfer(ModuleInfer):
    def infer(self, x: VarLenTensor) -> VarLenTensor:
        return self.mlp(x)
