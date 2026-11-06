# Copyright (c) Meta Platforms, Inc. and affiliates.
# See lightx2v/models/networks/vggt/LICENSE.txt.
# ViT layer portions retain the Apache-2.0 licensing of the source implementation.

import math
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.vggt.heads.head_act import activate_head, activate_pose
from lightx2v.models.networks.vggt.heads.utils import create_uv_grid, position_grid_to_embed


def _expand_tokens(tokens, batch, views):
    first = tokens[:, :1].expand(batch, 1, *tokens.shape[2:])
    others = tokens[:, 1:].expand(batch, views - 1, *tokens.shape[2:])
    return torch.cat((first, others), dim=1).reshape(batch * views, *tokens.shape[2:])


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulate the input tensor using scaling and shifting parameters."""
    return x * (1 + scale) + shift


def custom_interpolate(x: torch.Tensor, size: Tuple[int, int] = None, scale_factor: float = None, mode: str = "bilinear", align_corners: bool = True) -> torch.Tensor:
    """Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate."""
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=input_elements // INT_MAX + 1, dim=0)
        interpolated_chunks = [nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class VGGTTransformerInfer(BaseTransformerInfer):
    """VGGT computation with explicit checkpoint-weight arguments."""

    def _infer_blocks(self, weights, x):
        for block in weights:
            x = self.infer_block(block, x)
        return x

    def infer(self, weights, images):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        (tokens, patch_start) = self.infer_aggregator(weights.aggregator, images)
        poses = self.infer_camera(weights.camera_head, tokens)
        (depth, depth_conf) = self.infer_dpt(weights.depth_head, tokens, images=images, patch_start_idx=patch_start)
        (points, points_conf) = self.infer_dpt(weights.point_head, tokens, images=images, patch_start_idx=patch_start)
        return {"pose_enc": poses[-1], "depth": depth, "depth_conf": depth_conf, "world_points": points, "world_points_conf": points_conf}

    def infer_aggregator(self, weights, images):
        (batch, views, channels, height, width) = images.shape
        images = ((images - weights._resnet_mean) / weights._resnet_std).view(batch * views, channels, height, width)
        patches = self.infer_vision_transformer(weights.patch_embed, images)["x_norm_patchtokens"]
        camera = _expand_tokens(weights.camera_token, batch, views)
        registers = _expand_tokens(weights.register_token, batch, views)
        tokens = torch.cat((camera, registers, patches), dim=1)
        (count, dim) = tokens.shape[1:]
        positions = weights.position_getter(batch * views, height // weights.patch_size, width // weights.patch_size, images.device) + 1
        special = positions.new_zeros((batch * views, weights.patch_start_idx, 2))
        frame_positions = torch.cat((special, positions), dim=1)
        global_positions = frame_positions.view(batch, views * count, 2)
        output = []
        for index, (frame, global_block) in enumerate(zip(weights.frame_blocks, weights.global_blocks)):
            tokens = self.infer_block(frame, tokens.reshape(batch * views, count, dim), pos=frame_positions)
            frame_features = tokens.view(batch, views, count, dim)
            tokens = self.infer_block(global_block, tokens.reshape(batch, views * count, dim), pos=global_positions)
            output.append(torch.cat((frame_features, tokens.view(batch, views, count, dim)), dim=-1) if index in weights.cached_layers else None)
        return (output, weights.patch_start_idx)

    def _vision_transformer_interpolate_pos_encoding(self, weights, x, width, height):
        patches = x.shape[1] - 1
        if patches == weights.pos_embed.shape[1] - 1 and width == height:
            return weights.pos_embed
        positions = weights.pos_embed.float()
        side = int(math.sqrt(positions.shape[1] - 1))
        grid = positions[:, 1:].reshape(1, side, side, 1024).permute(0, 3, 1, 2)
        grid = nn.functional.interpolate(grid, size=(width // weights.patch_size, height // weights.patch_size), mode="bicubic", antialias=True)
        grid = grid.permute(0, 2, 3, 1).reshape(1, -1, 1024)
        return torch.cat((positions[:, :1], grid), dim=1).to(x.dtype)

    def infer_vision_transformer(self, weights, images):
        (_, _, width, height) = images.shape
        tokens = self.infer_patch_embed(weights.patch_embed, images)
        tokens = torch.cat((weights.cls_token.expand(images.shape[0], -1, -1), tokens), dim=1)
        tokens = tokens + self._vision_transformer_interpolate_pos_encoding(weights, tokens, width, height)
        tokens = torch.cat((tokens[:, :1], weights.register_tokens.expand(images.shape[0], -1, -1), tokens[:, 1:]), dim=1)
        for block in weights.blocks:
            tokens = self.infer_block(block, tokens)
        return {"x_norm_patchtokens": weights.norm(tokens)[:, 5:]}

    def infer_patch_embed(self, weights, x: Tensor) -> Tensor:
        (_, _, H, W) = x.shape
        (patch_H, patch_W) = weights.patch_size
        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"
        x = weights.proj(x)
        (H, W) = (x.size(2), x.size(3))
        x = x.flatten(2).transpose(1, 2)
        x = weights.norm(x)
        if not weights.flatten_embedding:
            x = x.reshape(-1, H, W, weights.embed_dim)
        return x

    def infer_block(self, weights, x, pos=None):
        x = x + self.infer_layer_scale(weights.ls1, self.infer_attention(weights.attn, weights.norm1(x), pos=pos))
        return x + self.infer_layer_scale(weights.ls2, self.infer_mlp(weights.mlp, weights.norm2(x)))

    def infer_attention(self, weights, x, pos=None):
        (batch, tokens, channels) = x.shape
        qkv = weights.qkv(x).reshape(batch, tokens, 3, weights.num_heads, weights.head_dim).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)
        (q, k) = (weights.q_norm(q), weights.k_norm(k))
        if weights.rope is not None:
            (q, k) = (self.infer_rope(weights.rope, q, pos), self.infer_rope(weights.rope, k, pos))
        out = weights.attention(q, k, v).transpose(1, 2).reshape(batch, tokens, channels)
        return weights.proj(out)

    def infer_mlp(self, weights, x: Tensor) -> Tensor:
        x = weights.fc1(x)
        x = weights.act(x)
        x = weights.drop(x)
        x = weights.fc2(x)
        x = weights.drop(x)
        return x

    def infer_layer_scale(self, weights, x: Tensor) -> Tensor:
        if not hasattr(weights, "gamma"):
            return weights(x)
        return x.mul_(weights.gamma) if weights.inplace else x * weights.gamma

    def _rope_compute_frequency_components(self, weights, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings."""
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in weights.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / weights.base_frequency**exponents
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            weights.frequency_cache[cache_key] = (cos_components, sin_components)
        return weights.frequency_cache[cache_key]

    def _rope_rotate_features(self, weights, x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions."""
        feature_dim = x.shape[-1]
        (x1, x2) = (x[..., : feature_dim // 2], x[..., feature_dim // 2 :])
        return torch.cat((-x2, x1), dim=-1)

    def _rope_apply_1d_rope(self, weights, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension."""
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return tokens * cos + self._rope_rotate_features(weights, tokens) * sin

    def infer_rope(self, weights, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens."""
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"
        feature_dim = tokens.size(-1) // 2
        max_position = int(positions.max()) + 1
        (cos_comp, sin_comp) = self._rope_compute_frequency_components(weights, feature_dim, max_position, tokens.device, tokens.dtype)
        (vertical_features, horizontal_features) = tokens.chunk(2, dim=-1)
        vertical_features = self._rope_apply_1d_rope(weights, vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._rope_apply_1d_rope(weights, horizontal_features, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical_features, horizontal_features), dim=-1)

    def infer_camera(self, weights, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """Forward pass to predict camera parameters."""
        tokens = aggregated_tokens_list[-1]
        pose_tokens = tokens[:, :, 0]
        pose_tokens = weights.token_norm(pose_tokens)
        pred_pose_enc_list = self._camera_trunk_fn(weights, pose_tokens, num_iterations)
        return pred_pose_enc_list

    def _camera_trunk_fn(self, weights, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """Iteratively refine camera pose predictions."""
        (B, S, C) = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []
        for _ in range(num_iterations):
            if pred_pose_enc is None:
                module_input = weights.embed_pose(weights.empty_pose_tokens.expand(B, S, -1))
            else:
                pred_pose_enc = pred_pose_enc.detach()
                module_input = weights.embed_pose(pred_pose_enc)
            (shift_msa, scale_msa, gate_msa) = weights.poseLN_modulation(module_input).chunk(3, dim=-1)
            pose_tokens_modulated = gate_msa * modulate(weights.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens
            pose_tokens_modulated = self._infer_blocks(weights.trunk, pose_tokens_modulated)
            pred_pose_enc_delta = self.infer_mlp(weights.pose_branch, weights.trunk_norm(pose_tokens_modulated))
            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta
            activated_pose = activate_pose(pred_pose_enc, trans_act=weights.trans_act, quat_act=weights.quat_act, fl_act=weights.fl_act)
            pred_pose_enc_list.append(activated_pose)
        return pred_pose_enc_list

    def infer_dpt(
        self, weights, aggregated_tokens_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frames_chunk_size: int = 8
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8."""
        (B, S, _, H, W) = images.shape
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._dpt_forward_impl(weights, aggregated_tokens_list, images, patch_start_idx)
        assert frames_chunk_size > 0
        all_preds = []
        all_conf = []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            if weights.feature_only:
                chunk_output = self._dpt_forward_impl(weights, aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx)
                all_preds.append(chunk_output)
            else:
                (chunk_preds, chunk_conf) = self._dpt_forward_impl(weights, aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx)
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)
        if weights.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return (torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1))

    def _dpt_forward_impl(
        self, weights, aggregated_tokens_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frames_start_idx: int = None, frames_end_idx: int = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Implementation of the forward pass through the DPT head."""
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
        (B, S, _, H, W) = images.shape
        (patch_h, patch_w) = (H // weights.patch_size, W // weights.patch_size)
        out = []
        dpt_idx = 0
        for layer_idx in weights.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            x = x.reshape(B * S, -1, x.shape[-1])
            x = weights.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = weights.projects[dpt_idx](x)
            if weights.pos_embed:
                x = self._dpt_apply_pos_embed(weights, x, W, H)
            x = weights.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1
        out = self._dpt_scratch_forward(weights, out)
        out = custom_interpolate(out, (int(patch_h * weights.patch_size / weights.down_ratio), int(patch_w * weights.patch_size / weights.down_ratio)), mode="bilinear", align_corners=True)
        if weights.pos_embed:
            out = self._dpt_apply_pos_embed(weights, out, W, H)
        if weights.feature_only:
            return out.view(B, S, *out.shape[1:])
        out = weights.scratch.output_conv2(out)
        (preds, conf) = activate_head(out, activation=weights.activation, conf_activation=weights.conf_activation)
        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return (preds, conf)

    def _dpt_apply_pos_embed(self, weights, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Apply positional embedding to tensor x."""
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def _dpt_scratch_forward(self, weights, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the fusion blocks."""
        (layer_1, layer_2, layer_3, layer_4) = features
        layer_1_rn = weights.scratch.layer1_rn(layer_1)
        layer_2_rn = weights.scratch.layer2_rn(layer_2)
        layer_3_rn = weights.scratch.layer3_rn(layer_3)
        layer_4_rn = weights.scratch.layer4_rn(layer_4)
        out = self._feature_fusion(weights.scratch.refinenet4, layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        out = self._feature_fusion(weights.scratch.refinenet3, out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        out = self._feature_fusion(weights.scratch.refinenet2, out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        out = self._feature_fusion(weights.scratch.refinenet1, out, layer_1_rn)
        del layer_1_rn, layer_1
        out = weights.scratch.output_conv1(out)
        return out

    def _residual_conv(self, weights, x):
        """Forward pass."""
        out = weights.activation(x)
        out = weights.conv1(out)
        if weights.norm1 is not None:
            out = weights.norm1(out)
        out = weights.activation(out)
        out = weights.conv2(out)
        if weights.norm2 is not None:
            out = weights.norm2(out)
        return weights.skip_add.add(out, x)

    def _feature_fusion(self, weights, *xs, size=None):
        """Forward pass."""
        output = xs[0]
        if weights.has_residual:
            res = self._residual_conv(weights.resConfUnit1, xs[1])
            output = weights.skip_add.add(output, res)
        output = self._residual_conv(weights.resConfUnit2, output)
        if size is None and weights.size is None:
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": weights.size}
        else:
            modifier = {"size": size}
        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=weights.align_corners)
        output = weights.out_conv(output)
        return output
