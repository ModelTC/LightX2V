from typing import Protocol, TypeVar

import torch
from einops import rearrange

from lightx2v.models.video_encoders.hf.ltx2.upsampler.pixel_shuffle import PixelShuffleND
from lightx2v.models.video_encoders.hf.ltx2.upsampler.res_block import ResBlock
from lightx2v.models.video_encoders.hf.ltx2.upsampler.spatial_rational_resampler import SpatialRationalResampler
from lightx2v.utils.ltx2_utils import *

ModelType = TypeVar("ModelType")


class ModelConfigurator(Protocol[ModelType]):
    """Protocol for model loader classes that instantiates models from a configuration dictionary."""

    @classmethod
    def from_config(cls, config: dict) -> ModelType: ...


class LatentUpsampler(torch.nn.Module):
    """
    Model to upsample VAE latents spatially and/or temporally.
    Args:
        in_channels (`int`): Number of channels in the input latent
        mid_channels (`int`): Number of channels in the middle layers
        num_blocks_per_stage (`int`): Number of ResBlocks to use in each stage (pre/post upsampling)
        dims (`int`): Number of dimensions for convolutions (2 or 3)
        spatial_upsample (`bool`): Whether to spatially upsample the latent
        temporal_upsample (`bool`): Whether to temporally upsample the latent
        spatial_scale (`float`): Scale factor for spatial upsampling
        rational_resampler (`bool`): Whether to use a rational resampler for spatial upsampling
    """

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        spatial_scale: float = 2.0,
        rational_resampler: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dims = dims
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample
        self.spatial_scale = float(spatial_scale)
        self.rational_resampler = rational_resampler

        conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d

        self.initial_conv = conv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = torch.nn.GroupNorm(32, mid_channels)
        self.initial_activation = torch.nn.SiLU()

        self.res_blocks = torch.nn.ModuleList([ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)])

        if spatial_upsample and temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(3),
            )
        elif spatial_upsample:
            if rational_resampler:
                self.upsampler = SpatialRationalResampler(mid_channels=mid_channels, scale=self.spatial_scale)
            else:
                self.upsampler = torch.nn.Sequential(
                    torch.nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
                    PixelShuffleND(2),
                )
        elif temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(1),
            )
        else:
            raise ValueError("Either spatial_upsample or temporal_upsample must be True")

        self.post_upsample_res_blocks = torch.nn.ModuleList([ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)])

        self.final_conv = conv(mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, _, f, _, _ = latent.shape

        if self.dims == 2:
            x = rearrange(latent, "b c f h w -> (b f) c h w")
            x = self.initial_conv(x)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            x = self.upsampler(x)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        else:
            x = self.initial_conv(latent)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            if self.temporal_upsample:
                x = self.upsampler(x)
                # remove the first frame after upsampling.
                # This is done because the first frame encodes one pixel frame.
                x = x[:, :, 1:, :, :]
            elif isinstance(self.upsampler, SpatialRationalResampler):
                x = self.upsampler(x)
            else:
                x = rearrange(x, "b c f h w -> (b f) c h w")
                x = self.upsampler(x)
                x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)

        return x


class LatentUpsamplerConfigurator(ModelConfigurator[LatentUpsampler]):
    """
    Configurator for LatentUpsampler model.
    Used to create a LatentUpsampler model from a configuration dictionary.
    """

    @classmethod
    def from_config(cls: type[LatentUpsampler], config: dict) -> LatentUpsampler:
        in_channels = config.get("in_channels", 128)
        mid_channels = config.get("mid_channels", 512)
        num_blocks_per_stage = config.get("num_blocks_per_stage", 4)
        dims = config.get("dims", 3)
        spatial_upsample = config.get("spatial_upsample", True)
        temporal_upsample = config.get("temporal_upsample", False)
        spatial_scale = config.get("spatial_scale", 2.0)
        rational_resampler = config.get("rational_resampler", False)
        return LatentUpsampler(
            in_channels=in_channels,
            mid_channels=mid_channels,
            num_blocks_per_stage=num_blocks_per_stage,
            dims=dims,
            spatial_upsample=spatial_upsample,
            temporal_upsample=temporal_upsample,
            spatial_scale=spatial_scale,
            rational_resampler=rational_resampler,
        )
