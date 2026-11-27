# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from typing import Tuple, Union  # noqa: UP035 - exported aliases used by the split infer modules

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    def __init__(self, num_feats: int, max_seq_len: Tensor | int, temperature: int = 10000, use_real: bool = False, theta_rescale_factor: float = 1.0, interpolation_factor: float = 1.0) -> None:
        super().__init__()
        assert num_feats % 2 == 0, "num_feats (head_dim) must be even for RoPE."
        self.num_feats = num_feats
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.use_real = use_real
        self.theta_rescale_factor = theta_rescale_factor
        self.interpolation_factor = interpolation_factor
        if isinstance(max_seq_len, int):
            max_seq_len = torch.arange(max_seq_len).float()
        if theta_rescale_factor != 1.0:
            temperature *= theta_rescale_factor ** (self.num_feats / (self.num_feats - 2))
        dim_t = torch.arange(0, self.num_feats, 2, dtype=torch.float32)
        freqs = 1.0 / temperature ** (2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_feats)
        freqs = torch.outer(max_seq_len.float() * interpolation_factor, freqs)
        if use_real:
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
            self.freqs_cis = (freqs_cos, freqs_sin)
        else:
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            self.freqs_cis = freqs_cis

    def apply_rotary_emb(self, xq: Tensor, xk: Tensor) -> tuple[Tensor, Tensor]:
        from lightx2v.models.networks.hymotion.infer.positional_encoding import RotaryEmbedding_apply_rotary_emb

        return RotaryEmbedding_apply_rotary_emb(self, xq, xk)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(num_feats={self.num_feats}, "
        repr_str += f"max_seq_len={self.max_seq_len}, "
        repr_str += f"temperature={self.temperature}, "
        repr_str += f"use_real={self.use_real}, "
        repr_str += f"theta_rescale_factor={self.theta_rescale_factor}, "
        repr_str += f"interpolation_factor={self.interpolation_factor})"
        return repr_str


__all__ = ["Tensor", "Tuple", "Union", "torch"]
