# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from lightx2v.common.ops.module_norm import LayerNorm32
from lightx2v.common.ops.sparse.attention import SparseMultiHeadAttention
from lightx2v.models.networks.trellis2.weights.layers.sparse_transformer.blocks import SparseFeedForwardNetWeights


class ModulatedSparseTransformerCrossBlockWeights(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """

    _inference_type = ("lightx2v.models.networks.trellis2.infer.layers.sparse_transformer.modulated", "ModulatedSparseTransformerCrossBlockInfer")

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-06)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-06)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-06)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(channels, ctx_channels=ctx_channels, num_heads=num_heads, type="cross", attn_mode="full", qkv_bias=qkv_bias, qk_rms_norm=qk_rms_norm_cross)
        self.mlp = SparseFeedForwardNetWeights(channels, mlp_ratio=mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True))
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels**0.5)

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.trellis2.infer.layers.sparse_transformer.modulated import ModulatedSparseTransformerCrossBlockInfer

        return ModulatedSparseTransformerCrossBlockInfer(self).infer(*args, **kwargs)
