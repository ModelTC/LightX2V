# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# See lightx2v/models/networks/vggt/LICENSE.txt.

import torch
import torch.nn as nn

from lightx2v.models.networks.vggt.weights.layers import MlpWeights
from lightx2v.models.networks.vggt.weights.layers.block import BlockWeights


class CameraHeadWeights(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    _inference_type = ("lightx2v.models.networks.vggt.infer.heads.camera_head", "CameraHeadInfer")

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
    ):
        super().__init__()
        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.trunk = nn.Sequential(*[BlockWeights(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values) for _ in range(trunk_depth)])
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-06)
        self.pose_branch = MlpWeights(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.vggt.infer.heads.camera_head import CameraHeadInfer

        return CameraHeadInfer(self).infer(*args, **kwargs)
