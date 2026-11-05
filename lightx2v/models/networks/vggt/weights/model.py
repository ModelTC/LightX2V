# Copyright (c) Meta Platforms, Inc. and affiliates.
# See LICENSE.txt in this directory.

from torch import nn

from lightx2v.models.networks.vggt.weights.aggregator import AggregatorWeights
from lightx2v.models.networks.vggt.weights.heads.camera_head import CameraHeadWeights
from lightx2v.models.networks.vggt.weights.heads.dpt_head import DPTHeadWeights


class VGGTWeights(nn.Module):
    """VGGT-1B camera, depth, and point-map inference."""

    _inference_type = ("lightx2v.models.networks.vggt.infer.model", "VGGTInfer")

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        self.aggregator = AggregatorWeights(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHeadWeights(dim_in=2 * embed_dim)
        self.point_head = DPTHeadWeights(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHeadWeights(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.vggt.infer.model import VGGTInfer

        return VGGTInfer(self).infer(*args, **kwargs)
