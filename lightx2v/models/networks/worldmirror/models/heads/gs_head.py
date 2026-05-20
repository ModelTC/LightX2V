from typing import List

import torch
import torch.nn as nn

from .dense_head import _BaseDPTHead


class GSFeatHead(_BaseDPTHead):
    """
    GS feature head that only outputs fused GS features.

    This head is used when gs depth is disabled. It skips the prediction
    conv (output_conv2) and returns only the fused GS feature map.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        pos_embed: bool = True,
        down_ratio: int = 1,
        gradient_checkpoint: bool = False,
    ) -> None:
        super().__init__(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            out_channels=out_channels,
            pos_embed=pos_embed,
            down_ratio=down_ratio,
            gradient_checkpoint=gradient_checkpoint,
            _cast_pos_embed_dtype=False,
        )
        conv2_in_channels = features // 2
        self.input_merger = nn.Sequential(
            nn.Conv2d(3, conv2_in_channels, 7, 1, 3),
            nn.ReLU(),
        )

    def forward(
        self,
        token_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> torch.Tensor:
        B, S, _, H, W = images.shape

        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(token_list, images, patch_start_idx)

        assert frames_chunk_size > 0
        gs_chunks = []
        for frame_start in range(0, S, frames_chunk_size):
            frame_end = min(frame_start + frames_chunk_size, S)
            gs = self._forward_impl(token_list, images, patch_start_idx, frame_start, frame_end)
            gs_chunks.append(gs)

        return torch.cat(gs_chunks, dim=1)

    def _forward_impl(
        self,
        token_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frame_start: int = None,
        frame_end: int = None,
    ) -> torch.Tensor:
        if frame_start is not None and frame_end is not None:
            images = images[:, frame_start:frame_end].contiguous()

        B, S, _, H, W = images.shape

        fused = self._extract_fused_features(token_list, B, S, H, W, patch_start_idx, frame_start, frame_end)

        img_flat = images.reshape(B * S, -1, H, W)
        img_feat = self.input_merger(img_flat)
        fused = fused + img_feat
        fused = fused.reshape(B, S, *fused.shape[1:])
        return fused
