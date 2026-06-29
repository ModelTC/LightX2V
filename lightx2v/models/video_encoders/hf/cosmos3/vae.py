import gc
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)

CACHE_T = 2


@dataclass
class DecoderOutput:
    sample: torch.Tensor


class AvgDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        x = x.view(
            b,
            c,
            t // self.factor_t,
            self.factor_t,
            h // self.factor_s,
            self.factor_s,
            w // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, c * self.factor, t // self.factor_t, h // self.factor_s, w // self.factor_s)
        x = x.view(b, self.out_channels, self.group_size, t // self.factor_t, h // self.factor_s, w // self.factor_s)
        return x.mean(dim=2)


class WanCausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        needs_fp32 = x.dtype in (torch.float16, torch.bfloat16) or any(t in str(x.dtype) for t in ("float4_", "float8_"))
        normalized = F.normalize(x.float() if needs_fp32 else x, dim=(1 if self.channel_first else -1)).to(x.dtype)
        return normalized * self.scale * self.gamma + self.bias


class WanUpsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class DupUp3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t, factor_s=1):
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(x.size(0), self.out_channels, x.size(2) * self.factor_t, x.size(4) * self.factor_s, x.size(6) * self.factor_s)
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class WanResample(nn.Module):
    def __init__(self, dim: int, mode: str, upsample_out_dim: int | None = None):
        super().__init__()
        self.mode = mode
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        if mode == "upsample2d":
            self.resample = nn.Sequential(WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, upsample_out_dim, 3, padding=1))
        elif mode == "upsample3d":
            self.resample = nn.Sequential(WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, upsample_out_dim, 3, padding=1))
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = WanCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                if cache_x.shape[2] < 2 and feat_cache[idx] == "Rep":
                    cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                x = self.time_conv(x) if feat_cache[idx] == "Rep" else self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), 3).reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], dim=2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x


class WanResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = WanRMSNorm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMSNorm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.conv_shortcut(x)
        x = F.silu(self.norm1(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        x = self.dropout(F.silu(self.norm2(x)))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)
        return x + h


class WanAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = WanRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity


class WanMidBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.resnets = nn.ModuleList([WanResidualBlock(dim, dim, dropout), WanResidualBlock(dim, dim, dropout)])
        self.attentions = nn.ModuleList([WanAttentionBlock(dim)])

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = self.attentions[0](x)
        x = self.resnets[1](x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x


class WanResidualDownBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, num_res_blocks: int, temperal_downsample=False, down_flag=False):
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.downsampler = None
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(out_dim, mode=mode)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x + self.avg_shortcut(x_copy)


class WanEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        temperal_downsample=[True, True, False],
        dropout=0.0,
        is_residual: bool = False,
    ):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if not is_residual:
                raise NotImplementedError("Cosmos3 VAE wrapper currently expects the Wan2.2 residual encoder.")
            self.down_blocks.append(
                WanResidualDownBlock(
                    in_dim,
                    out_dim,
                    dropout,
                    num_res_blocks,
                    temperal_downsample=temperal_downsample[i] if i != len(dim_mult) - 1 else False,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
        self.mid_block = WanMidBlock(out_dim, dropout)
        self.norm_out = WanRMSNorm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, z_dim, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)
        for layer in self.down_blocks:
            x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = F.silu(self.norm_out(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class WanResidualUpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, dropout: float = 0.0, temperal_upsample: bool = False, up_flag: bool = False):
        super().__init__()
        self.avg_shortcut = DupUp3D(in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2) if up_flag else None
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsampler = WanResample(out_dim, "upsample3d" if temperal_upsample else "upsample2d", upsample_out_dim=out_dim) if up_flag else None

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.upsampler is not None:
            x = self.upsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)
        return x


class WanDecoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        temperal_upsample=[False, True, True],
        dropout=0.0,
        out_channels: int = 3,
        is_residual: bool = False,
    ):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = WanMidBlock(dims[0], dropout)
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            up_flag = i != len(dim_mult) - 1
            if is_residual:
                up_block = WanResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                )
            else:
                raise NotImplementedError("Cosmos3 VAE wrapper currently expects the Wan2.2 residual decoder.")
            self.up_blocks.append(up_block)
        self.norm_out = WanRMSNorm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
        x = F.silu(self.norm_out(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


def patchify(x, patch_size):
    if patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Height ({height}) and width ({width}) must be divisible by patch_size ({patch_size})")
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.view(batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size)


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x
    batch_size, c_patches, frames, height, width = x.shape
    channels = c_patches // (patch_size * patch_size)
    x = x.view(batch_size, channels, patch_size, patch_size, frames, height, width)
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    return x.view(batch_size, channels, frames, height * patch_size, width * patch_size)


class AutoencoderKLWanDecodeOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.z_dim = config["z_dim"]
        self.patch_size = config.get("patch_size")
        self.post_quant_conv = WanCausalConv3d(self.z_dim, self.z_dim, 1)
        self.decoder = WanDecoder3d(
            dim=config.get("decoder_base_dim", config["base_dim"]),
            z_dim=self.z_dim,
            dim_mult=config["dim_mult"],
            num_res_blocks=config["num_res_blocks"],
            temperal_upsample=list(config["temperal_downsample"])[::-1],
            dropout=config.get("dropout", 0.0),
            out_channels=config["out_channels"],
            is_residual=config.get("is_residual", False),
        )
        self._conv_num = sum(isinstance(m, WanCausalConv3d) for m in self.decoder.modules())
        self.clear_cache()

    def clear_cache(self):
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        _, _, num_frame, _, _ = z.shape
        self.clear_cache()
        x = self.post_quant_conv(z)
        out = None
        for i in range(num_frame):
            self._conv_idx = [0]
            decoded = self.decoder(
                x[:, :, i : i + 1],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                first_chunk=(i == 0),
            )
            out = decoded if out is None else torch.cat([out, decoded], dim=2)
        if self.patch_size is not None:
            out = unpatchify(out, self.patch_size)
        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        if not return_dict:
            return (out,)
        return DecoderOutput(sample=out)


class AutoencoderKLWan(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.z_dim = config["z_dim"]
        self.patch_size = config.get("patch_size")
        self.encoder = WanEncoder3d(
            in_channels=config["in_channels"],
            dim=config["base_dim"],
            z_dim=self.z_dim * 2,
            dim_mult=config["dim_mult"],
            num_res_blocks=config["num_res_blocks"],
            temperal_downsample=config["temperal_downsample"],
            dropout=config.get("dropout", 0.0),
            is_residual=config.get("is_residual", False),
        )
        self.quant_conv = WanCausalConv3d(self.z_dim * 2, self.z_dim * 2, 1)
        self.post_quant_conv = WanCausalConv3d(self.z_dim, self.z_dim, 1)
        self.decoder = WanDecoder3d(
            dim=config.get("decoder_base_dim", config["base_dim"]),
            z_dim=self.z_dim,
            dim_mult=config["dim_mult"],
            num_res_blocks=config["num_res_blocks"],
            temperal_upsample=list(config["temperal_downsample"])[::-1],
            dropout=config.get("dropout", 0.0),
            out_channels=config["out_channels"],
            is_residual=config.get("is_residual", False),
        )
        self._enc_conv_num = sum(isinstance(m, WanCausalConv3d) for m in self.encoder.modules())
        self._conv_num = sum(isinstance(m, WanCausalConv3d) for m in self.decoder.modules())
        self.clear_cache()

    def clear_cache(self):
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num

    def _encode(self, x: torch.Tensor):
        _, _, num_frame, _, _ = x.shape
        self.clear_cache()
        if self.patch_size is not None:
            x = patchify(x, patch_size=self.patch_size)

        iter_ = 1 + (num_frame - 1) // 4
        out = None
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                encoded = self.encoder(x[:, :, :1], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                encoded = self.encoder(x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            out = encoded if out is None else torch.cat([out, encoded], dim=2)
        encoded = self.quant_conv(out)
        self.clear_cache()
        return encoded

    def encode(self, x: torch.Tensor):
        moments = self._encode(x)
        mean, _ = torch.chunk(moments, 2, dim=1)
        return mean

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        _, _, num_frame, _, _ = z.shape
        self.clear_cache()
        x = self.post_quant_conv(z)
        out = None
        for i in range(num_frame):
            self._conv_idx = [0]
            decoded = self.decoder(
                x[:, :, i : i + 1],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                first_chunk=(i == 0),
            )
            out = decoded if out is None else torch.cat([out, decoded], dim=2)
        if self.patch_size is not None:
            out = unpatchify(out, self.patch_size)
        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        if not return_dict:
            return (out,)
        return DecoderOutput(sample=out)


class Cosmos3WanVAE:
    def __init__(self, config):
        self.config = config
        self.cpu_offload = config.get("vae_cpu_offload", config.get("cpu_offload", False))
        self.device = torch.device("cpu") if self.cpu_offload else torch.device(AI_DEVICE)
        self.dtype = GET_DTYPE()
        self.load()

    def load(self):
        vae_path = self.config.get("vae_path", os.path.join(self.config["model_path"], "vae"))
        with open(os.path.join(vae_path, "config.json"), "r") as f:
            self.vae_config = json.load(f)
        self.load_encoder = self.config.get("task") == "i2v" or self.config.get("cosmos3_load_vae_encoder", False)
        model_cls = AutoencoderKLWan if self.load_encoder else AutoencoderKLWanDecodeOnly
        self.model = model_cls(self.vae_config).to(self.device).to(self.dtype)
        weight_path = os.path.join(vae_path, "diffusion_pytorch_model.safetensors")
        prefixes = ("decoder.", "post_quant_conv.")
        if self.load_encoder:
            prefixes = prefixes + ("encoder.", "quant_conv.")
        state = {}
        with safe_open(weight_path, framework="pt", device=str(self.device)) as f:
            for key in f.keys():
                if not key.startswith(prefixes):
                    continue
                tensor = f.get_tensor(key)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=self.dtype)
                state[key] = tensor
        self.model.load_state_dict(state, strict=False)
        self.model.eval().requires_grad_(False)
        self.latents_mean = torch.tensor(self.vae_config["latents_mean"], dtype=self.dtype).view(1, -1, 1, 1, 1)
        self.latents_std = torch.tensor(self.vae_config["latents_std"], dtype=self.dtype).view(1, -1, 1, 1, 1)

    @staticmethod
    def _to_pil_frames(video: torch.Tensor):
        frames = video[0].permute(1, 0, 2, 3)
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = frames.detach().cpu().float().permute(0, 2, 3, 1).numpy()
        return [Image.fromarray((frame * 255.0).round().astype(np.uint8)) for frame in frames]

    @torch.no_grad()
    def encode(self, video: torch.Tensor):
        if not self.load_encoder:
            raise RuntimeError("Cosmos3WanVAE was loaded without encoder. Set task=i2v or cosmos3_load_vae_encoder=True.")
        if self.cpu_offload:
            self.model.to(torch.device(AI_DEVICE))
        video = video.to(device=next(self.model.parameters()).device, dtype=self.dtype)
        raw_mu = self.model.encode(video)
        mean = self.latents_mean.to(raw_mu.device)
        std = self.latents_std.to(raw_mu.device)
        latents = ((raw_mu - mean) / std).to(video.dtype)
        if self.cpu_offload:
            latents = latents.to(AI_DEVICE)
            self.model.to(torch.device("cpu"))
            torch_device_module.empty_cache()
            gc.collect()
        return latents

    @torch.no_grad()
    def decode(self, latents, input_info):
        if self.cpu_offload:
            self.model.to(torch.device(AI_DEVICE))
        latents = latents.to(device=next(self.model.parameters()).device, dtype=self.dtype)
        mean = self.latents_mean.to(latents.device)
        std = self.latents_std.to(latents.device)
        z_raw = latents * std + mean
        decoded = self.model.decode(z_raw).sample
        if self.cpu_offload:
            decoded = decoded.cpu().float()
            self.model.to(torch.device("cpu"))
            torch_device_module.empty_cache()
            gc.collect()
        if input_info.return_result_tensor:
            return (decoded / 2 + 0.5).clamp(0, 1)
        return self._to_pil_frames(decoded)
