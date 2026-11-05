# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from lightx2v.models.networks.trellis2.weights.layers.attention import RotaryPositionEmbedderWeights
from lightx2v.models.networks.trellis2.weights.layers.transformer import AbsolutePositionEmbedderWeights, ModulatedTransformerCrossBlockWeights


class TimestepEmbedderWeights(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    _inference_type = ("lightx2v.models.networks.trellis2.infer.sparse_structure_flow", "TimestepEmbedderInfer")

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.trellis2.infer.sparse_structure_flow import TimestepEmbedderInfer

        return TimestepEmbedderInfer(self).infer(*args, **kwargs)


class SparseStructureFlowModelWeights(nn.Module):
    _inference_type = ("lightx2v.models.networks.trellis2.infer.sparse_structure_flow", "SparseStructureFlowModelInfer")

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
        **kwargs,
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
            pos_embedder = AbsolutePositionEmbedderWeights(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device="cpu") for res in [resolution] * 3], indexing="ij")
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb, persistent=False)
        elif pe_mode == "rope":
            pos_embedder = RotaryPositionEmbedderWeights(self.model_channels // self.num_heads, 3)
            coords = torch.meshgrid(*[torch.arange(res, device="cpu") for res in [resolution] * 3], indexing="ij")
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            rope_phases = pos_embedder(coords)
            self.register_buffer("rope_phases", rope_phases, persistent=False)
        if pe_mode != "rope":
            self.rope_phases = None
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlockWeights(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode="full",
                    use_rope=pe_mode == "rope",
                    rope_freq=rope_freq,
                    share_mod=share_mod,
                    qk_rms_norm=self.qk_rms_norm,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_layer = nn.Linear(model_channels, out_channels)

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.trellis2.infer.sparse_structure_flow import SparseStructureFlowModelInfer

        return SparseStructureFlowModelInfer(self).infer(*args, **kwargs)
