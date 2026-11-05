# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# See lightx2v/models/networks/vggt/LICENSE.txt.

import torch

from lightx2v.common.transformer_infer.module_infer import ModuleInfer
from lightx2v.models.networks.vggt.heads.head_act import activate_pose


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulate the input tensor using scaling and shifting parameters."""
    return x * (1 + scale) + shift


class CameraHeadInfer(ModuleInfer):
    def infer(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
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
