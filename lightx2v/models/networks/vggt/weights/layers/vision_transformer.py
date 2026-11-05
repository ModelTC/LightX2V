# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from functools import partial

import torch
from torch import nn

from lightx2v.models.networks.vggt.weights.layers.block import BlockWeights
from lightx2v.models.networks.vggt.weights.layers.patch_embed import PatchEmbedWeights


class DinoVisionTransformerWeights(nn.Module):
    """The ViT-L/14 patch encoder from the released VGGT-1B checkpoint."""

    _inference_type = ("lightx2v.models.networks.vggt.infer.layers.vision_transformer", "DinoVisionTransformerInfer")

    def __init__(self, img_size=518, patch_size=14):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedWeights(img_size=img_size, patch_size=patch_size, embed_dim=1024)
        self.cls_token = nn.Parameter(torch.empty(1, 1, 1024))
        self.pos_embed = nn.Parameter(torch.empty(1, self.patch_embed.num_patches + 1, 1024))
        self.register_tokens = nn.Parameter(torch.empty(1, 4, 1024))
        self.blocks = nn.ModuleList([BlockWeights(1024, num_heads=16, norm_layer=partial(nn.LayerNorm, eps=1e-06), init_values=1.0) for _ in range(24)])
        self.norm = nn.LayerNorm(1024, eps=1e-06)

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.vggt.infer.layers.vision_transformer import DinoVisionTransformerInfer

        return DinoVisionTransformerInfer(self).infer(*args, **kwargs)
