# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

import lightx2v.common.ops.sparse as sp
from lightx2v.models.networks.trellis2.weights.layers.sparse_transformer import ModulatedSparseTransformerCrossBlockWeights
from lightx2v.models.networks.trellis2.weights.layers.transformer import AbsolutePositionEmbedderWeights
from lightx2v.models.networks.trellis2.weights.sparse_structure_flow import TimestepEmbedderWeights


class SLatFlowModelWeights(nn.Module):
    _inference_type = ("lightx2v.models.networks.trellis2.infer.structured_latent_flow", "SLatFlowModelInfer")

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.bfloat16
        self.t_embedder = TimestepEmbedderWeights(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True))
        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedderWeights(model_channels)
        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                ModulatedSparseTransformerCrossBlockWeights(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode="full",
                    use_rope=pe_mode == "rope",
                    rope_freq=rope_freq,
                    share_mod=self.share_mod,
                    qk_rms_norm=self.qk_rms_norm,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_layer = sp.SparseLinear(model_channels, out_channels)

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.trellis2.infer.structured_latent_flow import SLatFlowModelInfer

        return SLatFlowModelInfer(self).infer(*args, **kwargs)
