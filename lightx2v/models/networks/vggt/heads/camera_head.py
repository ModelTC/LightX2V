# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# See lightx2v/models/networks/vggt/LICENSE.txt.

import torch
import torch.nn as nn

from lightx2v.models.networks.vggt.heads.head_act import activate_pose
from lightx2v.models.networks.vggt.layers import Mlp
from lightx2v.models.networks.vggt.layers.block import Block


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

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
        self.trunk = nn.Sequential(*[Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values) for _ in range(trunk_depth)])
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-06)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """Forward pass to predict camera parameters."""
        tokens = aggregated_tokens_list[-1]
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)
        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """Iteratively refine camera pose predictions."""
        (B, S, C) = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []
        for _ in range(num_iterations):
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)
            (shift_msa, scale_msa, gate_msa) = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens
            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))
            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta
            activated_pose = activate_pose(pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act)
            pred_pose_enc_list.append(activated_pose)
        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulate the input tensor using scaling and shifting parameters."""
    return x * (1 + scale) + shift
