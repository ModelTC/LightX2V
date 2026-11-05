# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# See the Apache-2.0 LICENSE in the repository root.
# Implementation of 2D Rotary Position Embeddings (RoPE).

from typing import Tuple

import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.module_infer import ModuleInfer


class RotaryPositionEmbedding2DInfer(ModuleInfer):
    def _compute_frequency_components(self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings."""
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / self.base_frequency**exponents
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)
        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions."""
        feature_dim = x.shape[-1]
        (x1, x2) = (x[..., : feature_dim // 2], x[..., feature_dim // 2 :])
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension."""
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return tokens * cos + self._rotate_features(tokens) * sin

    def infer(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens."""
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"
        feature_dim = tokens.size(-1) // 2
        max_position = int(positions.max()) + 1
        (cos_comp, sin_comp) = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)
        (vertical_features, horizontal_features) = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical_features, horizontal_features), dim=-1)
