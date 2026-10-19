# Copyright (c) Meta Platforms, Inc. and affiliates.
# See lightx2v/models/networks/vggt/LICENSE.txt.

import torch
from torch import nn

from .layers.block import Block
from .layers.rope import PositionGetter, RotaryPositionEmbedding2D
from .layers.vision_transformer import DinoVisionTransformer


class Aggregator(nn.Module):
    """Alternate frame and global attention, retaining the four DPT feature levels."""

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = DinoVisionTransformer(img_size=img_size, patch_size=patch_size)
        self.rope = RotaryPositionEmbedding2D(frequency=100)
        self.position_getter = PositionGetter()
        self.frame_blocks = nn.ModuleList([Block(embed_dim, num_heads=16, init_values=0.01, qk_norm=True, rope=self.rope) for _ in range(24)])
        self.global_blocks = nn.ModuleList([Block(embed_dim, num_heads=16, init_values=0.01, qk_norm=True, rope=self.rope) for _ in range(24)])
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, 4, embed_dim))
        self.patch_start_idx = 5
        self.cached_layers = {4, 11, 17, 23}
        for name, values in (("_resnet_mean", [0.485, 0.456, 0.406]), ("_resnet_std", [0.229, 0.224, 0.225])):
            self.register_buffer(name, torch.tensor(values).view(1, 1, 3, 1, 1), persistent=False)

    def forward(self, images):
        batch, views, channels, height, width = images.shape
        images = ((images - self._resnet_mean) / self._resnet_std).view(batch * views, channels, height, width)
        patches = self.patch_embed(images)["x_norm_patchtokens"]
        camera = _expand_tokens(self.camera_token, batch, views)
        registers = _expand_tokens(self.register_token, batch, views)
        tokens = torch.cat((camera, registers, patches), dim=1)
        count, dim = tokens.shape[1:]
        positions = self.position_getter(batch * views, height // self.patch_size, width // self.patch_size, images.device) + 1
        special = positions.new_zeros((batch * views, self.patch_start_idx, 2))
        frame_positions = torch.cat((special, positions), dim=1)
        global_positions = frame_positions.view(batch, views * count, 2)
        output = []
        for index, (frame, global_block) in enumerate(zip(self.frame_blocks, self.global_blocks)):
            tokens = frame(tokens.reshape(batch * views, count, dim), pos=frame_positions)
            frame_features = tokens.view(batch, views, count, dim)
            tokens = global_block(tokens.reshape(batch, views * count, dim), pos=global_positions)
            output.append(torch.cat((frame_features, tokens.view(batch, views, count, dim)), dim=-1) if index in self.cached_layers else None)
        return output, self.patch_start_idx


def _expand_tokens(tokens, batch, views):
    first = tokens[:, :1].expand(batch, 1, *tokens.shape[2:])
    others = tokens[:, 1:].expand(batch, views - 1, *tokens.shape[2:])
    return torch.cat((first, others), dim=1).reshape(batch * views, *tokens.shape[2:])
