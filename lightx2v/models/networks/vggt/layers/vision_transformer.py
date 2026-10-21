# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

import math
from functools import partial

import torch
from torch import nn

from .block import Block
from .patch_embed import PatchEmbed


class DinoVisionTransformer(nn.Module):
    """The ViT-L/14 patch encoder from the released VGGT-1B checkpoint."""

    def __init__(self, img_size=518, patch_size=14):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=1024)
        self.cls_token = nn.Parameter(torch.empty(1, 1, 1024))
        self.pos_embed = nn.Parameter(torch.empty(1, self.patch_embed.num_patches + 1, 1024))
        self.register_tokens = nn.Parameter(torch.empty(1, 4, 1024))
        self.blocks = nn.ModuleList([Block(1024, num_heads=16, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=1.0) for _ in range(24)])
        self.norm = nn.LayerNorm(1024, eps=1e-6)

    def interpolate_pos_encoding(self, x, width, height):
        patches = x.shape[1] - 1
        if patches == self.pos_embed.shape[1] - 1 and width == height:
            return self.pos_embed
        positions = self.pos_embed.float()
        side = int(math.sqrt(positions.shape[1] - 1))
        grid = positions[:, 1:].reshape(1, side, side, 1024).permute(0, 3, 1, 2)
        grid = nn.functional.interpolate(grid, size=(width // self.patch_size, height // self.patch_size), mode="bicubic", antialias=True)
        grid = grid.permute(0, 2, 3, 1).reshape(1, -1, 1024)
        return torch.cat((positions[:, :1], grid), dim=1).to(x.dtype)

    def forward(self, images):
        _, _, width, height = images.shape
        tokens = self.patch_embed(images)
        tokens = torch.cat((self.cls_token.expand(images.shape[0], -1, -1), tokens), dim=1)
        tokens = tokens + self.interpolate_pos_encoding(tokens, width, height)
        tokens = torch.cat((tokens[:, :1], self.register_tokens.expand(images.shape[0], -1, -1), tokens[:, 1:]), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return {"x_norm_patchtokens": self.norm(tokens)[:, 5:]}
