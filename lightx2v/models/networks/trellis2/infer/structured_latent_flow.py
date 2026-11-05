# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

from typing import List, Optional, Union

import torch
import torch.nn.functional as F

import lightx2v.common.ops.sparse as sp
from lightx2v.common.transformer_infer.module_infer import ModuleInfer


class SLatFlowModelInfer(ModuleInfer):
    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    def infer(self, x: sp.SparseTensor, t: torch.Tensor, cond: Union[torch.Tensor, List[torch.Tensor]], concat_cond: Optional[sp.SparseTensor] = None, **kwargs) -> sp.SparseTensor:
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)
        h = self.input_layer(x)
        h = h.to(dtype=self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.to(dtype=self.dtype)
        cond = cond.to(dtype=self.dtype)
        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + pe.to(dtype=self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        h = h.to(dtype=x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h
