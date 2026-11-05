# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

from typing import Union

import torch

from lightx2v.common.ops.sparse.basic import SparseTensor, VarLenTensor
from lightx2v.common.transformer_infer.module_infer import ModuleInfer


class ModulatedSparseTransformerCrossBlockInfer(ModuleInfer):
    def infer(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor]) -> SparseTensor:
        if self.share_mod:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x
