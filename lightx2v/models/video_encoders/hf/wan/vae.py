# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger

from lightx2v.utils.utils import load_weights

__all__ = [
    "WanVAE",
]

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[: c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2 :, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.spatial_compression_ratio = 2 ** len(self.temperal_downsample)

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        # modules
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def blend_v(self, a, b, blend_extent):
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a, b, blend_extent):
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x, scale):
        _, _, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                self.clear_cache()
                time = []
                frame_range = 1 + (num_frames - 1) // 4
                for k in range(frame_range):
                    self._enc_conv_idx = [0]
                    if k == 0:
                        tile = x[:, :, :1, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                    else:
                        tile = x[
                            :,
                            :,
                            1 + 4 * (k - 1) : 1 + 4 * k,
                            i : i + self.tile_sample_min_height,
                            j : j + self.tile_sample_min_width,
                        ]
                    tile = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                    mu, log_var = self.conv1(tile).chunk(2, dim=1)
                    if isinstance(scale[0], torch.Tensor):
                        mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
                    else:
                        mu = (mu - scale[0]) * scale[1]

                    time.append(mu)

                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(self, z, scale):
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]

        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                self.clear_cache()
                time = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                    tile = self.conv2(tile)
                    decoded = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx)
                    time.append(decoded)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        return dec

    def encode(self, x, scale):
        self.clear_cache()
        ## cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]

        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()

        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)

        self.clear_cache()
        return out

    def cached_decode(self, z, scale):
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, device="cpu", cpu_offload=False, dtype=torch.float, load_from_rank0=False, **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    cfg.update(**kwargs)

    # init model
    with torch.device("meta"):
        model = WanVAE_(**cfg)

    # load checkpoint
    weights_dict = load_weights(pretrained_path, cpu_offload=cpu_offload, load_from_rank0=load_from_rank0)
    for k in weights_dict.keys():
        if weights_dict[k].dtype != dtype:
            weights_dict[k] = weights_dict[k].to(dtype)
    model.load_state_dict(weights_dict, assign=True)

    return model


class WanVAE:
    def __init__(
        self,
        z_dim=16,
        vae_pth="cache/vae_step_411000.pth",
        dtype=torch.float,
        device="cuda",
        parallel=False,
        use_tiling=False,
        cpu_offload=False,
        use_2d_split=True,
        load_from_rank0=False,
    ):
        self.dtype = dtype
        self.device = device
        self.parallel = parallel
        self.use_tiling = use_tiling
        self.cpu_offload = cpu_offload
        self.use_2d_split = use_2d_split

        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.inv_std = 1.0 / torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, self.inv_std]

        # (height, width, world_size) -> (world_size_h, world_size_w)
        self.grid_table = {
            # world_size = 2
            (60, 104, 2): (1, 2),
            (68, 120, 2): (1, 2),
            (90, 160, 2): (1, 2),
            (60, 60, 2): (1, 2),
            (72, 72, 2): (1, 2),
            (88, 88, 2): (1, 2),
            (120, 120, 2): (1, 2),
            (104, 60, 2): (2, 1),
            (120, 68, 2): (2, 1),
            (160, 90, 2): (2, 1),
            # world_size = 4
            (60, 104, 4): (2, 2),
            (68, 120, 4): (2, 2),
            (90, 160, 4): (2, 2),
            (60, 60, 4): (2, 2),
            (72, 72, 4): (2, 2),
            (88, 88, 4): (2, 2),
            (120, 120, 4): (2, 2),
            (104, 60, 4): (2, 2),
            (120, 68, 4): (2, 2),
            (160, 90, 4): (2, 2),
            # world_size = 8
            (60, 104, 8): (2, 4),
            (68, 120, 8): (2, 4),
            (90, 160, 8): (2, 4),
            (60, 60, 8): (2, 4),
            (72, 72, 8): (2, 4),
            (88, 88, 8): (2, 4),
            (120, 120, 8): (2, 4),
            (104, 60, 8): (4, 2),
            (120, 68, 8): (4, 2),
            (160, 90, 8): (4, 2),
        }

        # init model
        self.model = _video_vae(pretrained_path=vae_pth, z_dim=z_dim, cpu_offload=cpu_offload, dtype=dtype, load_from_rank0=load_from_rank0).eval().requires_grad_(False).to(device).to(dtype)

    def _calculate_2d_grid(self, latent_height, latent_width, world_size):
        if (latent_height, latent_width, world_size) in self.grid_table:
            best_h, best_w = self.grid_table[(latent_height, latent_width, world_size)]
            # logger.info(f"Vae using cached 2D grid: {best_h}x{best_w} grid for {latent_height}x{latent_width} latent")
            return best_h, best_w

        best_h, best_w = 1, world_size
        min_aspect_diff = float("inf")

        for h in range(1, world_size + 1):
            if world_size % h == 0:
                w = world_size // h
                if latent_height % h == 0 and latent_width % w == 0:
                    # Calculate how close this grid is to square
                    aspect_diff = abs((latent_height / h) - (latent_width / w))
                    if aspect_diff < min_aspect_diff:
                        min_aspect_diff = aspect_diff
                        best_h, best_w = h, w
        # logger.info(f"Vae using 2D grid & Update cache: {best_h}x{best_w} grid for {latent_height}x{latent_width} latent")
        self.grid_table[(latent_height, latent_width, world_size)] = (best_h, best_w)
        return best_h, best_w

    def current_device(self):
        return next(self.model.parameters()).device

    def to_cpu(self):
        self.model.encoder = self.model.encoder.to("cpu")
        self.model.decoder = self.model.decoder.to("cpu")
        self.model = self.model.to("cpu")
        self.mean = self.mean.cpu()
        self.inv_std = self.inv_std.cpu()
        self.scale = [self.mean, self.inv_std]

    def to_cuda(self):
        self.model.encoder = self.model.encoder.to("cuda")
        self.model.decoder = self.model.decoder.to("cuda")
        self.model = self.model.to("cuda")
        self.mean = self.mean.cuda()
        self.inv_std = self.inv_std.cuda()
        self.scale = [self.mean, self.inv_std]

    def encode_dist(self, video, world_size, cur_rank, split_dim):
        spatial_ratio = 8

        if split_dim == 3:
            total_latent_len = video.shape[3] // spatial_ratio
        elif split_dim == 4:
            total_latent_len = video.shape[4] // spatial_ratio
        else:
            raise ValueError(f"Unsupported split_dim: {split_dim}")

        splited_chunk_len = total_latent_len // world_size
        padding_size = 1

        video_chunk_len = splited_chunk_len * spatial_ratio
        video_padding_len = padding_size * spatial_ratio

        if cur_rank == 0:
            if split_dim == 3:
                video_chunk = video[:, :, :, : video_chunk_len + 2 * video_padding_len, :].contiguous()
            elif split_dim == 4:
                video_chunk = video[:, :, :, :, : video_chunk_len + 2 * video_padding_len].contiguous()
        elif cur_rank == world_size - 1:
            if split_dim == 3:
                video_chunk = video[:, :, :, -(video_chunk_len + 2 * video_padding_len) :, :].contiguous()
            elif split_dim == 4:
                video_chunk = video[:, :, :, :, -(video_chunk_len + 2 * video_padding_len) :].contiguous()
        else:
            start_idx = cur_rank * video_chunk_len - video_padding_len
            end_idx = (cur_rank + 1) * video_chunk_len + video_padding_len
            if split_dim == 3:
                video_chunk = video[:, :, :, start_idx:end_idx, :].contiguous()
            elif split_dim == 4:
                video_chunk = video[:, :, :, :, start_idx:end_idx].contiguous()

        if self.use_tiling:
            encoded_chunk = self.model.tiled_encode(video_chunk, self.scale)
        else:
            encoded_chunk = self.model.encode(video_chunk, self.scale)

        if cur_rank == 0:
            if split_dim == 3:
                encoded_chunk = encoded_chunk[:, :, :, :splited_chunk_len, :].contiguous()
            elif split_dim == 4:
                encoded_chunk = encoded_chunk[:, :, :, :, :splited_chunk_len].contiguous()
        elif cur_rank == world_size - 1:
            if split_dim == 3:
                encoded_chunk = encoded_chunk[:, :, :, -splited_chunk_len:, :].contiguous()
            elif split_dim == 4:
                encoded_chunk = encoded_chunk[:, :, :, :, -splited_chunk_len:].contiguous()
        else:
            if split_dim == 3:
                encoded_chunk = encoded_chunk[:, :, :, padding_size:-padding_size, :].contiguous()
            elif split_dim == 4:
                encoded_chunk = encoded_chunk[:, :, :, :, padding_size:-padding_size].contiguous()

        full_encoded = [torch.empty_like(encoded_chunk) for _ in range(world_size)]
        dist.all_gather(full_encoded, encoded_chunk)

        torch.cuda.synchronize()

        encoded = torch.cat(full_encoded, dim=split_dim)

        return encoded.squeeze(0)

    def encode_dist_2d(self, video, world_size_h, world_size_w, cur_rank_h, cur_rank_w):
        spatial_ratio = 8

        # Calculate chunk sizes for both dimensions
        total_latent_h = video.shape[3] // spatial_ratio
        total_latent_w = video.shape[4] // spatial_ratio

        chunk_h = total_latent_h // world_size_h
        chunk_w = total_latent_w // world_size_w

        padding_size = 1
        video_chunk_h = chunk_h * spatial_ratio
        video_chunk_w = chunk_w * spatial_ratio
        video_padding_h = padding_size * spatial_ratio
        video_padding_w = padding_size * spatial_ratio

        # Calculate H dimension slice
        if cur_rank_h == 0:
            h_start = 0
            h_end = video_chunk_h + 2 * video_padding_h
        elif cur_rank_h == world_size_h - 1:
            h_start = video.shape[3] - (video_chunk_h + 2 * video_padding_h)
            h_end = video.shape[3]
        else:
            h_start = cur_rank_h * video_chunk_h - video_padding_h
            h_end = (cur_rank_h + 1) * video_chunk_h + video_padding_h

        # Calculate W dimension slice
        if cur_rank_w == 0:
            w_start = 0
            w_end = video_chunk_w + 2 * video_padding_w
        elif cur_rank_w == world_size_w - 1:
            w_start = video.shape[4] - (video_chunk_w + 2 * video_padding_w)
            w_end = video.shape[4]
        else:
            w_start = cur_rank_w * video_chunk_w - video_padding_w
            w_end = (cur_rank_w + 1) * video_chunk_w + video_padding_w

        # Extract the video chunk for this process
        video_chunk = video[:, :, :, h_start:h_end, w_start:w_end].contiguous()

        # Encode the chunk
        if self.use_tiling:
            encoded_chunk = self.model.tiled_encode(video_chunk, self.scale)
        else:
            encoded_chunk = self.model.encode(video_chunk, self.scale)

        # Remove padding from encoded chunk
        if cur_rank_h == 0:
            encoded_h_start = 0
            encoded_h_end = chunk_h
        elif cur_rank_h == world_size_h - 1:
            encoded_h_start = encoded_chunk.shape[3] - chunk_h
            encoded_h_end = encoded_chunk.shape[3]
        else:
            encoded_h_start = padding_size
            encoded_h_end = encoded_chunk.shape[3] - padding_size

        if cur_rank_w == 0:
            encoded_w_start = 0
            encoded_w_end = chunk_w
        elif cur_rank_w == world_size_w - 1:
            encoded_w_start = encoded_chunk.shape[4] - chunk_w
            encoded_w_end = encoded_chunk.shape[4]
        else:
            encoded_w_start = padding_size
            encoded_w_end = encoded_chunk.shape[4] - padding_size

        encoded_chunk = encoded_chunk[:, :, :, encoded_h_start:encoded_h_end, encoded_w_start:encoded_w_end].contiguous()

        # Gather all chunks
        total_processes = world_size_h * world_size_w
        full_encoded = [torch.empty_like(encoded_chunk) for _ in range(total_processes)]

        dist.all_gather(full_encoded, encoded_chunk)

        torch.cuda.synchronize()

        # Reconstruct the full encoded tensor
        encoded_rows = []
        for h_idx in range(world_size_h):
            encoded_cols = []
            for w_idx in range(world_size_w):
                process_idx = h_idx * world_size_w + w_idx
                encoded_cols.append(full_encoded[process_idx])
            encoded_rows.append(torch.cat(encoded_cols, dim=4))

        encoded = torch.cat(encoded_rows, dim=3)

        return encoded.squeeze(0)

    def encode(self, video):
        """
        video: one video  with shape [1, C, T, H, W].
        """
        if self.cpu_offload:
            self.to_cuda()

        if self.parallel:
            world_size = dist.get_world_size()
            cur_rank = dist.get_rank()
            height, width = video.shape[3], video.shape[4]

            if self.use_2d_split:
                world_size_h, world_size_w = self._calculate_2d_grid(height // 8, width // 8, world_size)
                cur_rank_h = cur_rank // world_size_w
                cur_rank_w = cur_rank % world_size_w
                out = self.encode_dist_2d(video, world_size_h, world_size_w, cur_rank_h, cur_rank_w)
            else:
                # Original 1D splitting logic
                if width % world_size == 0:
                    out = self.encode_dist(video, world_size, cur_rank, split_dim=4)
                elif height % world_size == 0:
                    out = self.encode_dist(video, world_size, cur_rank, split_dim=3)
                else:
                    logger.info("Fall back to naive encode mode")
                    if self.use_tiling:
                        out = self.model.tiled_encode(video, self.scale).squeeze(0)
                    else:
                        out = self.model.encode(video, self.scale).squeeze(0)
        else:
            if self.use_tiling:
                out = self.model.tiled_encode(video, self.scale).squeeze(0)
            else:
                out = self.model.encode(video, self.scale).squeeze(0)

        if self.cpu_offload:
            self.to_cpu()
        return out

    def decode_dist(self, zs, world_size, cur_rank, split_dim):
        splited_total_len = zs.shape[split_dim]
        splited_chunk_len = splited_total_len // world_size
        padding_size = 1

        if cur_rank == 0:
            if split_dim == 2:
                zs = zs[:, :, : splited_chunk_len + 2 * padding_size, :].contiguous()
            elif split_dim == 3:
                zs = zs[:, :, :, : splited_chunk_len + 2 * padding_size].contiguous()
        elif cur_rank == world_size - 1:
            if split_dim == 2:
                zs = zs[:, :, -(splited_chunk_len + 2 * padding_size) :, :].contiguous()
            elif split_dim == 3:
                zs = zs[:, :, :, -(splited_chunk_len + 2 * padding_size) :].contiguous()
        else:
            if split_dim == 2:
                zs = zs[:, :, cur_rank * splited_chunk_len - padding_size : (cur_rank + 1) * splited_chunk_len + padding_size, :].contiguous()
            elif split_dim == 3:
                zs = zs[:, :, :, cur_rank * splited_chunk_len - padding_size : (cur_rank + 1) * splited_chunk_len + padding_size].contiguous()

        decode_func = self.model.tiled_decode if self.use_tiling else self.model.decode
        images = decode_func(zs.unsqueeze(0), self.scale).clamp_(-1, 1)

        if cur_rank == 0:
            if split_dim == 2:
                images = images[:, :, :, : splited_chunk_len * 8, :].contiguous()
            elif split_dim == 3:
                images = images[:, :, :, :, : splited_chunk_len * 8].contiguous()
        elif cur_rank == world_size - 1:
            if split_dim == 2:
                images = images[:, :, :, -splited_chunk_len * 8 :, :].contiguous()
            elif split_dim == 3:
                images = images[:, :, :, :, -splited_chunk_len * 8 :].contiguous()
        else:
            if split_dim == 2:
                images = images[:, :, :, 8 * padding_size : -8 * padding_size, :].contiguous()
            elif split_dim == 3:
                images = images[:, :, :, :, 8 * padding_size : -8 * padding_size].contiguous()

        full_images = [torch.empty_like(images) for _ in range(world_size)]
        dist.all_gather(full_images, images)

        torch.cuda.synchronize()

        images = torch.cat(full_images, dim=split_dim + 1)

        return images

    def decode_dist_2d(self, zs, world_size_h, world_size_w, cur_rank_h, cur_rank_w):
        total_h = zs.shape[2]
        total_w = zs.shape[3]

        chunk_h = total_h // world_size_h
        chunk_w = total_w // world_size_w

        padding_size = 1

        # Calculate H dimension slice
        if cur_rank_h == 0:
            h_start = 0
            h_end = chunk_h + 2 * padding_size
        elif cur_rank_h == world_size_h - 1:
            h_start = total_h - (chunk_h + 2 * padding_size)
            h_end = total_h
        else:
            h_start = cur_rank_h * chunk_h - padding_size
            h_end = (cur_rank_h + 1) * chunk_h + padding_size

        # Calculate W dimension slice
        if cur_rank_w == 0:
            w_start = 0
            w_end = chunk_w + 2 * padding_size
        elif cur_rank_w == world_size_w - 1:
            w_start = total_w - (chunk_w + 2 * padding_size)
            w_end = total_w
        else:
            w_start = cur_rank_w * chunk_w - padding_size
            w_end = (cur_rank_w + 1) * chunk_w + padding_size

        # Extract the latent chunk for this process
        zs_chunk = zs[:, :, h_start:h_end, w_start:w_end].contiguous()

        # Decode the chunk
        decode_func = self.model.tiled_decode if self.use_tiling else self.model.decode
        images_chunk = decode_func(zs_chunk.unsqueeze(0), self.scale).clamp_(-1, 1)

        # Remove padding from decoded chunk
        spatial_ratio = 8
        if cur_rank_h == 0:
            decoded_h_start = 0
            decoded_h_end = chunk_h * spatial_ratio
        elif cur_rank_h == world_size_h - 1:
            decoded_h_start = images_chunk.shape[3] - chunk_h * spatial_ratio
            decoded_h_end = images_chunk.shape[3]
        else:
            decoded_h_start = padding_size * spatial_ratio
            decoded_h_end = images_chunk.shape[3] - padding_size * spatial_ratio

        if cur_rank_w == 0:
            decoded_w_start = 0
            decoded_w_end = chunk_w * spatial_ratio
        elif cur_rank_w == world_size_w - 1:
            decoded_w_start = images_chunk.shape[4] - chunk_w * spatial_ratio
            decoded_w_end = images_chunk.shape[4]
        else:
            decoded_w_start = padding_size * spatial_ratio
            decoded_w_end = images_chunk.shape[4] - padding_size * spatial_ratio

        images_chunk = images_chunk[:, :, :, decoded_h_start:decoded_h_end, decoded_w_start:decoded_w_end].contiguous()

        # Gather all chunks
        total_processes = world_size_h * world_size_w
        full_images = [torch.empty_like(images_chunk) for _ in range(total_processes)]

        dist.all_gather(full_images, images_chunk)

        torch.cuda.synchronize()

        # Reconstruct the full image tensor
        image_rows = []
        for h_idx in range(world_size_h):
            image_cols = []
            for w_idx in range(world_size_w):
                process_idx = h_idx * world_size_w + w_idx
                image_cols.append(full_images[process_idx])
            image_rows.append(torch.cat(image_cols, dim=4))

        images = torch.cat(image_rows, dim=3)

        return images

    def decode(self, zs):
        if self.cpu_offload:
            self.to_cuda()

        if self.parallel:
            world_size = dist.get_world_size()
            cur_rank = dist.get_rank()
            latent_height, latent_width = zs.shape[2], zs.shape[3]

            if self.use_2d_split:
                world_size_h, world_size_w = self._calculate_2d_grid(latent_height, latent_width, world_size)
                cur_rank_h = cur_rank // world_size_w
                cur_rank_w = cur_rank % world_size_w
                images = self.decode_dist_2d(zs, world_size_h, world_size_w, cur_rank_h, cur_rank_w)
            else:
                # Original 1D splitting logic
                if latent_width % world_size == 0:
                    images = self.decode_dist(zs, world_size, cur_rank, split_dim=3)
                elif latent_height % world_size == 0:
                    images = self.decode_dist(zs, world_size, cur_rank, split_dim=2)
                else:
                    logger.info("Fall back to naive decode mode")
                    images = self.model.decode(zs.unsqueeze(0), self.scale).clamp_(-1, 1)
        else:
            decode_func = self.model.tiled_decode if self.use_tiling else self.model.decode
            images = decode_func(zs.unsqueeze(0), self.scale).clamp_(-1, 1)

        if self.cpu_offload:
            images = images.cpu()
            self.to_cpu()

        return images


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    # # Test both 1D and 2D splitting
    # print(f"Rank {dist.get_rank()}: Testing 1D splitting")
    # model_1d = WanVAE(vae_pth="/data/nvme0/models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth", dtype=torch.bfloat16, parallel=True, use_2d_split=False)
    # model_1d.to_cuda()

    input_tensor = torch.randn(1, 3, 17, 480, 480).to(torch.bfloat16).to("cuda")
    # encoded_tensor_1d = model_1d.encode(input_tensor)
    # print(f"rank {dist.get_rank()} 1D encoded_tensor shape: {encoded_tensor_1d.shape}")
    # decoded_tensor_1d = model_1d.decode(encoded_tensor_1d)
    # print(f"rank {dist.get_rank()} 1D decoded_tensor shape: {decoded_tensor_1d.shape}")

    print(f"Rank {dist.get_rank()}: Testing 2D splitting")
    model_2d = WanVAE(vae_pth="/data/nvme0/models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth", dtype=torch.bfloat16, parallel=True, use_2d_split=True)
    model_2d.to_cuda()

    encoded_tensor_2d = model_2d.encode(input_tensor)
    print(f"rank {dist.get_rank()} 2D encoded_tensor shape: {encoded_tensor_2d.shape}")
    decoded_tensor_2d = model_2d.decode(encoded_tensor_2d)
    print(f"rank {dist.get_rank()} 2D decoded_tensor shape: {decoded_tensor_2d.shape}")

    # # Verify that both methods produce the same results
    # if dist.get_rank() == 0:
    #     print(f"Encoded tensors match: {torch.allclose(encoded_tensor_1d, encoded_tensor_2d, atol=1e-5)}")
    #     print(f"Decoded tensors match: {torch.allclose(decoded_tensor_1d, decoded_tensor_2d, atol=1e-5)}")

    dist.destroy_process_group()
