# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

import torch

from lightx2v.common.transformer_infer.module_infer import ModuleInfer


class AbsolutePositionEmbedderInfer(ModuleInfer):
    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal position embeddings."""
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
        x (torch.Tensor): (N, D) tensor of spatial positions"""
        (N, D) = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed


class FeedForwardNetInfer(ModuleInfer):
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
