# Copyright (c) Meta Platforms, Inc. and affiliates.
# See lightx2v/models/networks/vggt/LICENSE.txt.

import torch
from torch import nn

from lightx2v.models.networks.vggt.weights.layers.block import BlockWeights
from lightx2v.models.networks.vggt.weights.layers.rope import PositionGetterWeights, RotaryPositionEmbedding2DWeights
from lightx2v.models.networks.vggt.weights.layers.vision_transformer import DinoVisionTransformerWeights


class AggregatorWeights(nn.Module):
    """Alternate frame and global attention, retaining the four DPT feature levels."""

    _inference_type = ("lightx2v.models.networks.vggt.infer.aggregator", "AggregatorInfer")

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = DinoVisionTransformerWeights(img_size=img_size, patch_size=patch_size)
        self.rope = RotaryPositionEmbedding2DWeights(frequency=100)
        self.position_getter = PositionGetterWeights()
        self.frame_blocks = nn.ModuleList([BlockWeights(embed_dim, num_heads=16, init_values=0.01, qk_norm=True, rope=self.rope) for _ in range(24)])
        self.global_blocks = nn.ModuleList([BlockWeights(embed_dim, num_heads=16, init_values=0.01, qk_norm=True, rope=self.rope) for _ in range(24)])
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, 4, embed_dim))
        self.patch_start_idx = 5
        self.cached_layers = {4, 11, 17, 23}
        for name, values in (("_resnet_mean", [0.485, 0.456, 0.406]), ("_resnet_std", [0.229, 0.224, 0.225])):
            self.register_buffer(name, torch.tensor(values).view(1, 1, 3, 1, 1), persistent=False)

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.vggt.infer.aggregator import AggregatorInfer

        return AggregatorInfer(self).infer(*args, **kwargs)
