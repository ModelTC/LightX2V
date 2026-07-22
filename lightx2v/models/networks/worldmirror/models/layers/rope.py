# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightx2v.common.ops.rope import TorchRealRope

_WORLDMIRROR_ROPE = TorchRealRope(layout="split_half")


@dataclass(frozen=True)
class RotaryPositionEmbedding2DCache:
    """Per-forward, position-gathered RoPE values for both spatial axes."""

    vertical_cos: torch.Tensor
    vertical_sin: torch.Tensor
    horizontal_cos: torch.Tensor
    horizontal_sin: torch.Tensor

    def reshape(self, batch_size: int, token_count: int) -> "RotaryPositionEmbedding2DCache":
        def reshape_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.shape[0] * tensor.shape[2] != batch_size * token_count:
                raise ValueError(
                    f"Cannot reshape RoPE cache from batch/tokens={tensor.shape[0]}/{tensor.shape[2]} "
                    f"to {batch_size}/{token_count}."
                )
            return tensor.reshape(batch_size, 1, token_count, tensor.shape[-1])

        return RotaryPositionEmbedding2DCache(
            vertical_cos=reshape_tensor(self.vertical_cos),
            vertical_sin=reshape_tensor(self.vertical_sin),
            horizontal_cos=reshape_tensor(self.horizontal_cos),
            horizontal_sin=reshape_tensor(self.horizontal_sin),
        )

    def select_batch(self, indices: torch.Tensor) -> "RotaryPositionEmbedding2DCache":
        return RotaryPositionEmbedding2DCache(
            vertical_cos=self.vertical_cos.index_select(0, indices),
            vertical_sin=self.vertical_sin.index_select(0, indices),
            horizontal_cos=self.horizontal_cos.index_select(0, indices),
            horizontal_sin=self.horizontal_sin.index_select(0, indices),
        )


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        cache_key = (height, width, torch.device(device))
        if cache_key not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[cache_key] = positions

        cached_positions = self.position_cache[cache_key]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1)


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(
        self,
        frequency: float = 100.0,
        scaling_factor: float = 1.0,
    ):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    def _apply_1d_rope(self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        return _WORLDMIRROR_ROPE.apply_single(tokens, (cos, sin))

    def prepare_cache(self, positions: torch.Tensor, head_dim: int, dtype: torch.dtype) -> RotaryPositionEmbedding2DCache:
        """Gather all position-dependent values once for one model forward."""
        if head_dim % 4 != 0:
            raise ValueError(f"2D RoPE head_dim must be divisible by 4, got {head_dim}.")
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("Positions must have shape (batch_size, n_tokens, 2).")

        feature_dim = head_dim // 2
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, positions.device, dtype)

        vertical_positions = positions[..., 0]
        horizontal_positions = positions[..., 1]
        vertical_cos = F.embedding(vertical_positions, cos_comp)[:, None, :, :]
        vertical_sin = F.embedding(vertical_positions, sin_comp)[:, None, :, :]
        horizontal_cos = F.embedding(horizontal_positions, cos_comp)[:, None, :, :]
        horizontal_sin = F.embedding(horizontal_positions, sin_comp)[:, None, :, :]
        return RotaryPositionEmbedding2DCache(vertical_cos, vertical_sin, horizontal_cos, horizontal_sin)

    @staticmethod
    def _validate_cache(tokens: torch.Tensor, cache: RotaryPositionEmbedding2DCache) -> None:
        if tokens.shape[0] != cache.vertical_cos.shape[0] or tokens.shape[-2] != cache.vertical_cos.shape[-2]:
            raise ValueError(
                f"RoPE cache shape does not match tokens: cache={cache.vertical_cos.shape}, tokens={tokens.shape}."
            )

    def apply_single(self, tokens: torch.Tensor, cache: RotaryPositionEmbedding2DCache) -> torch.Tensor:
        self._validate_cache(tokens, cache)
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = _WORLDMIRROR_ROPE.apply_single(vertical_features, (cache.vertical_cos, cache.vertical_sin))
        horizontal_features = _WORLDMIRROR_ROPE.apply_single(horizontal_features, (cache.horizontal_cos, cache.horizontal_sin))
        return torch.cat((vertical_features, horizontal_features), dim=-1)

    def apply_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cache: RotaryPositionEmbedding2DCache,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_cache(q, cache)
        self._validate_cache(k, cache)
        q_vertical, q_horizontal = q.chunk(2, dim=-1)
        k_vertical, k_horizontal = k.chunk(2, dim=-1)
        q_vertical, k_vertical = _WORLDMIRROR_ROPE.apply(
            q_vertical,
            k_vertical,
            (cache.vertical_cos, cache.vertical_sin),
        )
        q_horizontal, k_horizontal = _WORLDMIRROR_ROPE.apply(
            q_horizontal,
            k_horizontal,
            (cache.horizontal_cos, cache.horizontal_sin),
        )
        return torch.cat((q_vertical, q_horizontal), dim=-1), torch.cat((k_vertical, k_horizontal), dim=-1)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        cache = self.prepare_cache(positions, head_dim=tokens.shape[-1], dtype=tokens.dtype)
        return self.apply_single(tokens, cache)
