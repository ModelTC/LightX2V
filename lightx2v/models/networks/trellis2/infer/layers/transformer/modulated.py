# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

from typing import Optional

import torch

from lightx2v.common.transformer_infer.module_infer import ModuleInfer


class ModulatedTransformerCrossBlockInfer(ModuleInfer):
    def infer(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.share_mod:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h, phases=phases)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x
