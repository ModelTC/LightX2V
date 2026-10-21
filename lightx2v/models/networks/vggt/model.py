# Copyright (c) Meta Platforms, Inc. and affiliates.
# See LICENSE.txt in this directory.

from torch import nn

from .aggregator import Aggregator
from .heads.camera_head import CameraHead
from .heads.dpt_head import DPTHead


class VGGT(nn.Module):
    """VGGT-1B camera, depth, and point-map inference."""

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")

    def forward(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        tokens, patch_start = self.aggregator(images)
        poses = self.camera_head(tokens)
        depth, depth_conf = self.depth_head(tokens, images=images, patch_start_idx=patch_start)
        points, points_conf = self.point_head(tokens, images=images, patch_start_idx=patch_start)
        return {"pose_enc": poses[-1], "depth": depth, "depth_conf": depth_conf, "world_points": points, "world_points_conf": points_conf}
