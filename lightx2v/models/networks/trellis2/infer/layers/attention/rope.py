# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

import torch

from lightx2v.common.transformer_infer.module_infer import ModuleInfer


class RotaryPositionEmbedderInfer(ModuleInfer):
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    @staticmethod
    def apply_rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases.unsqueeze(-2)
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed

    def infer(self, indices: torch.Tensor) -> torch.Tensor:
        """Args:
        indices (torch.Tensor): [..., N, C] tensor of spatial positions"""
        assert indices.shape[-1] == self.dim, f"Last dim of indices must be {self.dim}"
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[-1] < self.head_dim // 2:
            padn = self.head_dim // 2 - phases.shape[-1]
            phases = torch.cat([phases, torch.polar(torch.ones(*phases.shape[:-1], padn, device=phases.device), torch.zeros(*phases.shape[:-1], padn, device=phases.device))], dim=-1)
        return phases
