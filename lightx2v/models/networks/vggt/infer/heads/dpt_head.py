# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# See lightx2v/models/networks/vggt/LICENSE.txt.
# Inspired by https://github.com/DepthAnything/Depth-Anything-V2

from typing import List, Tuple, Union

import torch
import torch.nn as nn

from lightx2v.common.transformer_infer.module_infer import ModuleInfer
from lightx2v.models.networks.vggt.heads.head_act import activate_head
from lightx2v.models.networks.vggt.heads.utils import create_uv_grid, position_grid_to_embed


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


class DPTHeadInfer(ModuleInfer):
    def infer(self, aggregated_tokens_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frames_chunk_size: int = 8) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)
        assert frames_chunk_size > 0
        all_preds = []
        all_conf = []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            if self.feature_only:
                chunk_output = self._forward_impl(aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx)
                all_preds.append(chunk_output)
            else:
                (chunk_preds, chunk_conf) = self._forward_impl(aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx)
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)
        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return (torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1))

    def _forward_impl(
        self, aggregated_tokens_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frames_start_idx: int = None, frames_end_idx: int = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Implementation of the forward pass through the DPT head."""
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
        (B, S, _, H, W) = images.shape
        (patch_h, patch_w) = (H // self.patch_size, W // self.patch_size)
        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            x = x.reshape(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1
        out = self.scratch_forward(out)
        out = custom_interpolate(out, (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)), mode="bilinear", align_corners=True)
        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)
        if self.feature_only:
            return out.view(B, S, *out.shape[1:])
        out = self.scratch.output_conv2(out)
        (preds, conf) = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)
        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return (preds, conf)

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Apply positional embedding to tensor x."""
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the fusion blocks."""
        (layer_1, layer_2, layer_3, layer_4) = features
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1
        out = self.scratch.output_conv1(out)
        return out


class ResidualConvUnitInfer(ModuleInfer):
    def infer(self, x):
        """Forward pass."""
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlockInfer(ModuleInfer):
    def infer(self, *xs, size=None):
        """Forward pass."""
        output = xs[0]
        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if size is None and self.size is None:
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output
