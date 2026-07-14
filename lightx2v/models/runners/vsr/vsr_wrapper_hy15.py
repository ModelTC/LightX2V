from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from einops import rearrange
from torch import Tensor

from lightx2v.models.video_encoders.hf.hunyuanvideo15.hunyuanvideo_15_vae import (
    CausalConv3d,
    RMS_norm,
    ResnetBlock,
    forward_with_checkpointing,
    swish,
)


class UpsamplerType(Enum):
    LEARNED = "learned"
    FIXED = "fixed"
    NONE = "none"
    LEARNED_FIXED = "learned_fixed"


@dataclass
class UpsamplerConfig:
    load_from: str
    enable: bool = False
    hidden_channels: int = 128
    num_blocks: int = 16
    model_type: UpsamplerType = UpsamplerType.NONE
    version: str = "720p"


class SRResidualCausalBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            CausalConv3d(channels, channels, kernel_size=3),
            nn.SiLU(inplace=True),
            CausalConv3d(channels, channels, kernel_size=3),
            nn.SiLU(inplace=True),
            CausalConv3d(channels, channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SRModel3DV2(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        num_blocks: int = 6,
        global_residual: bool = False,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = 64
        self.in_conv = CausalConv3d(in_channels, hidden_channels, kernel_size=3)
        self.blocks = nn.ModuleList([SRResidualCausalBlock3D(hidden_channels) for _ in range(num_blocks)])
        self.out_conv = CausalConv3d(hidden_channels, out_channels, kernel_size=3)
        self.global_residual = bool(global_residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.in_conv(x)
        for blk in self.blocks:
            y = blk(y)
        y = self.out_conv(y)
        if self.global_residual and (y.shape == residual.shape):
            y = y + residual
        return y


class Upsampler(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        num_res_blocks: int = 2,
        is_residual: bool = False,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.block_out_channels = block_out_channels
        self.z_channels = z_channels

        # assert block_out_channels[0] % z_channels == 0
        block_in = block_out_channels[0]
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3)

        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block

            self.up.append(up)

        self.norm_out = RMS_norm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3)

        self.gradient_checkpointing = False
        self.is_residual = is_residual

    def forward(self, z: Tensor, target_shape: Sequence[int] = None) -> Tensor:
        """
        Args:
            z: (B, C, T, H, W)
            target_shape: (H, W)
        """
        use_checkpointing = bool(self.training and self.gradient_checkpointing)
        if target_shape is not None and z.shape[-2:] != target_shape:
            bsz = z.shape[0]
            z = rearrange(z, "b c f h w -> (b f) c h w")
            z = F.interpolate(z, size=target_shape, mode="bilinear", align_corners=False)
            z = rearrange(z, "(b f) c h w -> b c f h w", b=bsz)

        # z to block_in
        repeats = self.block_out_channels[0] // (self.z_channels)
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        # upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = forward_with_checkpointing(
                    self.up[i_level].block[i_block],
                    h,
                    use_checkpointing=use_checkpointing,
                )
            if hasattr(self.up[i_level], "upsample"):
                h = forward_with_checkpointing(self.up[i_level].upsample, h, use_checkpointing=use_checkpointing)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h
