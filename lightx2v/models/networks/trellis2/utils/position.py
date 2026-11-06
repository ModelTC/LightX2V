# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

from typing import Tuple

import torch
from torch import nn


class AbsolutePositionEmbedder(nn.Module):
    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / 10000**self.freqs

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal position embeddings."""
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
        x (torch.Tensor): (N, D) tensor of spatial positions"""
        (N, D) = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, head_dim: int, dim: int = 3, rope_freq: Tuple[float, float] = (1.0, 10000.0)):
        super().__init__()
        assert head_dim % 2 == 0, "Head dim must be divisible by 2"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32, device="cpu") / self.freq_dim
        self.freqs = rope_freq[0] / rope_freq[1] ** self.freqs

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

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Args:
        indices (torch.Tensor): [..., N, C] tensor of spatial positions"""
        assert indices.shape[-1] == self.dim, f"Last dim of indices must be {self.dim}"
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[-1] < self.head_dim // 2:
            padn = self.head_dim // 2 - phases.shape[-1]
            phases = torch.cat([phases, torch.polar(torch.ones(*phases.shape[:-1], padn, device=phases.device), torch.zeros(*phases.shape[:-1], padn, device=phases.device))], dim=-1)
        return phases
