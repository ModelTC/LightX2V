# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightx2v.common.ops.sparse as sp
from lightx2v.common.ops.module_norm import LayerNorm32


class SparseResBlockC2S3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None, pred_subdiv: bool = True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.pred_subdiv = pred_subdiv
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-06)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-06)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        self.conv2 = sp.SparseConv3d(self.out_channels, self.out_channels, 3)
        self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        if pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseChannel2Spatial(2)

    def forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return (h, subdiv)
        else:
            return h


class SparseConvNeXtBlock3d(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.channels = channels
        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-06)
        self.conv = sp.SparseConv3d(channels, channels, 3)
        self.mlp = nn.Sequential(nn.Linear(channels, int(channels * mlp_ratio)), nn.SiLU(), nn.Linear(int(channels * mlp_ratio), channels))

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.conv(x)
        h = h.replace(self.norm(h.feats))
        h = h.replace(self.mlp(h.feats))
        return h + x


class SparseUnetVaeDecoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """

    def __init__(
        self,
        out_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.pred_subdiv = pred_subdiv
        self.dtype = torch.bfloat16
        self.output_layer = sp.SparseLinear(model_channels[-1], out_channels)
        self.from_latent = sp.SparseLinear(latent_channels, model_channels[0])
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(globals()[block_type[i]](model_channels[i], **block_args[i]))
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(globals()[up_block_type[i]](model_channels[i], model_channels[i + 1], pred_subdiv=pred_subdiv, **block_args[i]))

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    def forward(self, x: sp.SparseTensor, guide_subs: Optional[List[sp.SparseTensor]] = None, return_subs: bool = False) -> sp.SparseTensor:
        assert guide_subs is None or not self.pred_subdiv, "Only decoders with pred_subdiv=False can be used with guide_subs"
        assert not return_subs or self.pred_subdiv, "Only decoders with pred_subdiv=True can be used with return_subs"
        h = self.from_latent(x)
        h = h.type(self.dtype)
        subs = []
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if self.pred_subdiv:
                        (h, sub) = block(h)
                        subs.append(sub)
                    else:
                        h = block(h, subdiv=guide_subs[i] if guide_subs is not None else None)
                else:
                    h = block(h)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.output_layer(h)
        if return_subs:
            return (h, subs)
        else:
            return h
