# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# Copyright 2026 The LightX2V Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native MiniMax-H3 video VAE encoder/decoder.

The architecture and decode recipe are ported from the MiniMax-H3 implementation
pinned with the released checkpoint.  This module intentionally depends only on
PyTorch and safetensors: no third-party model implementation is imported at
runtime.

There are two explicit decode boundaries:

* :meth:`decode_raw` consumes already denormalized VAE latents and returns
  ImageNet-normalized RGB, matching the low-level autoencoder output.
* :meth:`decode` consumes diffusion-space latents, applies the released
  per-channel mean/std, runs the FP16-autocast decode with FP32 weights, and
  returns RGB in ``[0, 1]`` with shape ``[batch, 3, frames, height, width]``.
"""

from __future__ import annotations

import gc
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightx2v.models.video_encoders.hf.minimax_h3.weights import (
    SafetensorsSubsetReport,
    load_safetensors_subset,
)
from lightx2v_platform.base.global_var import AI_DEVICE

MINIMAX_H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_PIXEL_STD = (0.229, 0.224, 0.225)


def _empty_device_cache(device: torch.device) -> None:
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "empty_cache"):
        backend.empty_cache()


def _component_dir(model_path: str | Path, component: str) -> Path:
    model_path = Path(model_path)
    nested = model_path / component
    if nested.is_dir():
        return nested
    if model_path.name == component and model_path.is_dir():
        return model_path
    raise FileNotFoundError(f"Cannot find MiniMax-H3 {component!r} below {model_path}")


class _SwiGLU(nn.Module):
    """Checkpoint-compatible SwiGLU used by the ViT decoder."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.silu(gate)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, bias: bool = True) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        # Keep the original ``net.0.proj`` and ``net.2`` parameter names.
        self.net = nn.ModuleList(
            [
                _SwiGLU(dim, inner_dim, bias=bias),
                nn.Dropout(0.0),
                nn.Linear(inner_dim, dim, bias=bias),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class MiniMaxH3VideoCausalConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, spatial_padding=0, temporal_padding=0, spatial_padding_mode="reflect"):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, hidden_states):
        if self.spatial_padding > 0:
            p = self.spatial_padding
            hidden_states = F.pad(hidden_states, (p, p, p, p, 0, 0), mode=self.spatial_padding_mode)
        if self.temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0))
        return F.conv3d(hidden_states, self.weight, self.bias, stride=self.stride, dilation=self.dilation)


class MiniMaxH3VideoGroupNorm(nn.GroupNorm):
    def forward(self, hidden_states):
        batch, channels, frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous().view(batch * frames, channels, 1, height, width)
        hidden_states = super().forward(hidden_states)
        return hidden_states.view(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


class MiniMaxH3VideoResnetBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups=32, norm_eps=1e-6, spatial_padding_mode="reflect"):
        super().__init__()
        self.norm1 = MiniMaxH3VideoGroupNorm(norm_num_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = MiniMaxH3VideoCausalConv3d(in_channels, out_channels, 3, spatial_padding=1, temporal_padding=2, spatial_padding_mode=spatial_padding_mode)
        self.norm2 = MiniMaxH3VideoGroupNorm(norm_num_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = MiniMaxH3VideoCausalConv3d(out_channels, out_channels, 3, spatial_padding=1, temporal_padding=2, spatial_padding_mode=spatial_padding_mode)
        self.conv_shortcut = MiniMaxH3VideoCausalConv3d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))
        hidden_states = self.conv2(F.silu(self.norm2(hidden_states)))
        return hidden_states + (self.conv_shortcut(residual) if self.conv_shortcut is not None else residual)


class MiniMaxH3VideoDownsample3d(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_stride=1, spatial_stride=2, spatial_padding_mode="reflect"):
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = MiniMaxH3VideoCausalConv3d(in_channels, out_channels, 3, stride=(temporal_stride, spatial_stride, spatial_stride), temporal_padding=2, spatial_padding_mode=spatial_padding_mode)

    def forward(self, hidden_states):
        if self.spatial_stride == 2:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode)
        return self.conv(hidden_states)


class MiniMaxH3VideoDownBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, temporal_downsample_factor, spatial_downsample_factor, norm_num_groups=32, norm_eps=1e-6, spatial_padding_mode="reflect"):
        super().__init__()
        self.resnets = nn.ModuleList(
            [MiniMaxH3VideoResnetBlock3d(in_channels if index == 0 else out_channels, out_channels, norm_num_groups, norm_eps, spatial_padding_mode) for index in range(num_layers)]
        )
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = nn.ModuleList([MiniMaxH3VideoDownsample3d(out_channels, out_channels, temporal_downsample_factor, spatial_downsample_factor, spatial_padding_mode)])

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class MiniMaxH3VideoEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=48,
        block_out_channels=(128, 256, 256, 512, 512, 1024),
        layers_per_block=2,
        spatial_downsample_factors=(2, 2, 2, 2, 1, 1),
        temporal_downsample_factors=(1, 2, 2, 1, 1, 1),
        norm_num_groups=32,
        norm_eps=1e-6,
        spatial_padding_mode="reflect",
    ):
        super().__init__()
        self.conv_in = MiniMaxH3VideoCausalConv3d(
            in_channels,
            block_out_channels[0],
            3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoDownBlock3d(
                    block_in_channels[index],
                    block_out_channels[index],
                    layers_per_block,
                    temporal_downsample_factors[index],
                    spatial_downsample_factors[index],
                    norm_num_groups,
                    norm_eps,
                    spatial_padding_mode,
                )
                for index in range(len(block_out_channels))
            ]
        )
        self.norm_out = MiniMaxH3VideoGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps, affine=True)
        self.conv_out = MiniMaxH3VideoCausalConv3d(block_out_channels[-1], out_channels, 3, spatial_padding=1, temporal_padding=2, spatial_padding_mode=spatial_padding_mode)

    def forward(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)
        for block in self.down_blocks:
            hidden_states = block(hidden_states)
        return self.conv_out(F.silu(self.norm_out(hidden_states)))


class MiniMaxH3VideoRotaryPosEmbed(nn.Module):
    """Three-axis rotary embedding used by the non-causal ViT decoder."""

    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3) -> None:
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(f"dim={dim} must be divisible by 2 * num_axes={2 * num_axes}")
        self.dim = dim
        self.theta = theta
        self.num_axes = num_axes

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / self.theta ** torch.arange(
            0,
            1,
            2 * self.num_axes / self.dim,
            dtype=torch.float32,
            device=position_ids.device,
        )
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


class MiniMaxH3VideoAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=bias), nn.Dropout(0.0)])

    @staticmethod
    def _apply_rotary(
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        rotary_dim = cos.shape[-1]
        rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat([-second, first], dim=-1)
        return torch.cat([rotary * cos + rotated * sin, passthrough], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states).unflatten(2, (self.heads, self.dim_head))
        key = self.to_k(hidden_states).unflatten(2, (self.heads, self.dim_head))
        value = self.to_v(hidden_states).unflatten(2, (self.heads, self.dim_head))

        # The released implementation always normalizes Q/K in float32.
        query = self.norm_q(query.float()).to(query.dtype)
        key = self.norm_k(key.float()).to(key.dtype)

        if rotary_emb is not None:
            cos, sin = (value.to(query.dtype) for value in rotary_emb)
            query = self._apply_rotary(query, cos, sin)
            key = self._apply_rotary(key, cos, sin)

        # PyTorch SDPA uses [B, H, S, D]; the released attention dispatcher
        # presents [B, S, H, D] at its boundary.
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        return self.to_out[0](hidden_states.flatten(2, 3))


class MiniMaxH3VideoTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = MiniMaxH3VideoAttention(dim=dim, heads=heads, dim_head=dim_head, eps=eps, bias=bias)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = _FeedForward(dim, mult=ffn_mult, bias=bias)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states.float()).to(hidden_states.dtype)
        hidden_states = hidden_states + self.attn(norm_hidden_states, rotary_emb) * self.scale1
        norm_hidden_states = self.norm2(hidden_states.float()).to(hidden_states.dtype)
        return hidden_states + self.ff(norm_hidden_states) * self.scale2


class MiniMaxH3VideoViTDecoder3d(nn.Module):
    """Non-causal ViT decoder with register and zero class tokens."""

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        patch_size_t: int = 4,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.75,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.rope = MiniMaxH3VideoRotaryPosEmbed(int(attention_head_dim * rope_dim_ratio), theta=rope_theta)
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoTransformerBlock(
                    dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(batch_size, num_frames * height * width, num_channels)
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]

        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0 for size in (num_frames, height, width)]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        rotary_emb = self.rope(torch.cat([position_ids, suffix_ids], dim=1))

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.proj_out(self.norm_out(hidden_states))[:, :num_patches, :]
        patch_size, patch_size_t = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            patch_size_t,
            patch_size,
            patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )


class MiniMaxH3VideoVAE(nn.Module):
    """H3 video VAE that loads encoder and decoder from the original shards."""

    def __init__(
        self,
        config: dict,
        *,
        device: str | torch.device | None = None,
        cpu_offload: bool = False,
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.execution_device = torch.device(device or AI_DEVICE)
        self.cpu_offload = cpu_offload

        latent_channels = int(config.get("latent_channels", 24))
        out_channels = int(config.get("out_channels", 3))
        spatial_factors = tuple(int(value) for value in config.get("spatial_downsample_factors", (2, 2, 2, 2, 1, 1)))
        temporal_factors = tuple(int(value) for value in config.get("temporal_downsample_factors", (1, 2, 2, 1, 1, 1)))
        self.spatial_compression_ratio = math.prod(spatial_factors)
        self.temporal_compression_ratio = math.prod(temporal_factors)

        self.encoder = MiniMaxH3VideoEncoder3d(
            in_channels=int(config.get("in_channels", 3)),
            out_channels=2 * latent_channels,
            block_out_channels=tuple(config.get("block_out_channels", (128, 256, 256, 512, 512, 1024))),
            layers_per_block=int(config.get("layers_per_block", 2)),
            spatial_downsample_factors=spatial_factors,
            temporal_downsample_factors=temporal_factors,
            norm_num_groups=int(config.get("norm_num_groups", 32)),
            norm_eps=float(config.get("norm_eps", 1e-6)),
            spatial_padding_mode=config.get("spatial_padding_mode", "reflect"),
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)

        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            patch_size=self.spatial_compression_ratio,
            patch_size_t=self.temporal_compression_ratio,
            num_layers=int(config.get("decoder_num_layers", 36)),
            num_attention_heads=int(config.get("decoder_num_attention_heads", 32)),
            attention_head_dim=int(config.get("decoder_attention_head_dim", 64)),
            num_register_tokens=int(config.get("decoder_num_register_tokens", 4)),
            ffn_mult=int(config.get("decoder_ffn_mult", 4)),
            rope_theta=float(config.get("decoder_rope_theta", 100.0)),
            rope_dim_ratio=float(config.get("decoder_rope_dim_ratio", 0.75)),
            norm_eps=float(config.get("decoder_norm_eps", 1e-5)),
        )

        self.clip_length = int(config.get("clip_length", 17))
        self.token_drop = int(config.get("token_drop", 3))
        self.frame_pre_padding = (-self.clip_length) % self.temporal_compression_ratio
        self.tokens_chunk_size = math.ceil(self.clip_length / self.temporal_compression_ratio)
        self.token_overlap = (-self.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(
            self.token_overlap * self.temporal_compression_ratio - self.frame_pre_padding,
            0,
        )

        self.use_slicing = False
        self.use_tiling = True
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

        self.register_buffer("latents_mean", torch.empty(latent_channels), persistent=False)
        self.register_buffer("latents_std", torch.empty(latent_channels), persistent=False)
        self.register_buffer("pixel_mean", torch.empty(3), persistent=False)
        self.register_buffer("pixel_std", torch.empty(3), persistent=False)
        self._reset_runtime_buffers()
        self.load_report: SafetensorsSubsetReport | None = None

    def _reset_runtime_buffers(self) -> None:
        latent_channels = self.post_quant_conv.in_channels
        mean = self.config.get("latents_mean", [0.0] * latent_channels)
        std = self.config.get("latents_std", [1.0] * latent_channels)
        if len(mean) != latent_channels or len(std) != latent_channels:
            raise ValueError(f"Video latent statistics must contain {latent_channels} values, got mean={len(mean)}, std={len(std)}")
        self._buffers["latents_mean"] = torch.tensor(mean, dtype=torch.float32)
        self._buffers["latents_std"] = torch.tensor(std, dtype=torch.float32)
        self._buffers["pixel_mean"] = torch.tensor(MINIMAX_H3_PIXEL_MEAN, dtype=torch.float32)
        self._buffers["pixel_std"] = torch.tensor(MINIMAX_H3_PIXEL_STD, dtype=torch.float32)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device | None = None,
        cpu_offload: bool = False,
    ) -> "MiniMaxH3VideoVAE":
        vae_dir = _component_dir(model_path, "vae")
        with (vae_dir / "config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        # The released decoder is several GiB.  Constructing it on meta avoids
        # allocating and then immediately overwriting random initialized weights.
        with torch.device("meta"):
            model = cls(config, device=device, cpu_offload=cpu_offload)
        model._reset_runtime_buffers()
        model.load_report = load_safetensors_subset(model, vae_dir)
        model.eval().requires_grad_(False)
        if not cpu_offload:
            model.to(model.execution_device)
        return model

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_overlap_height: int | None = None,
        tile_sample_min_overlap_width: int | None = None,
    ) -> None:
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_overlap_height = tile_sample_min_overlap_height or self.tile_sample_min_overlap_height
        self.tile_sample_min_overlap_width = tile_sample_min_overlap_width or self.tile_sample_min_overlap_width

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def _split_tiles(self, length: int, tile_size: int, min_overlap: int) -> tuple[list[int], list[int], list[int]]:
        if tile_size >= length:
            return [0], [length], []

        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1

        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for index in range(remaining // self.spatial_compression_ratio):
            overlaps[index % (num_tiles - 1)] += self.spatial_compression_ratio

        starts = [0]
        for index in range(num_tiles - 1):
            starts.append(starts[-1] + tile_size - overlaps[index])
        return starts, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        if blend_extent <= 0:
            return b
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        result_rows = []
        for row_index, row in enumerate(tiles):
            result_row = []
            for column_index, tile in enumerate(row):
                if row_index > 0:
                    tile = self._blend(
                        tiles[row_index - 1][column_index],
                        tile,
                        height_overlaps[row_index - 1],
                        dim=-2,
                    )
                if column_index > 0:
                    tile = self._blend(row[column_index - 1], tile, width_overlaps[column_index - 1], dim=-1)
                if row_index < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[row_index], :]
                if column_index < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[column_index]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _decode_clip(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(latents))

        height = latents.shape[-2] * self.spatial_compression_ratio
        width = latents.shape[-1] * self.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = self._split_tiles(height, self.tile_sample_min_height, self.tile_sample_min_overlap_height)
        x_indices, x_lengths, x_overlaps = self._split_tiles(width, self.tile_sample_min_width, self.tile_sample_min_overlap_width)

        ratio = self.spatial_compression_ratio
        rows = []
        for y_position, y_length in zip(y_indices, y_lengths):
            row = []
            for x_position, x_length in zip(x_indices, x_lengths):
                tile = latents[
                    ...,
                    y_position // ratio : y_position // ratio + y_length // ratio,
                    x_position // ratio : x_position // ratio + x_length // ratio,
                ]
                row.append(self.decoder(self.post_quant_conv(tile)))
            rows.append(row)
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    def _encode_clip(self, pixels: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.quant_conv(self.encoder(pixels))
        height, width = pixels.shape[-2:]
        y_indices, y_lengths, y_overlaps = self._split_tiles(height, self.tile_sample_min_height, self.tile_sample_min_overlap_height)
        x_indices, x_lengths, x_overlaps = self._split_tiles(width, self.tile_sample_min_width, self.tile_sample_min_overlap_width)
        rows = []
        for y_position, y_length in zip(y_indices, y_lengths):
            row = []
            for x_position, x_length in zip(x_indices, x_lengths):
                tile = pixels[..., y_position : y_position + y_length, x_position : x_position + x_length]
                row.append(self.quant_conv(self.encoder(tile)))
            rows.append(row)
        latent_y_overlaps = [value // self.spatial_compression_ratio for value in y_overlaps]
        latent_x_overlaps = [value // self.spatial_compression_ratio for value in x_overlaps]
        return self._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        num_frames = pixels.shape[2]
        if num_frames % self.clip_length:
            pixels = torch.cat((pixels, pixels[:, :, -1:].repeat(1, 1, (-num_frames) % self.clip_length, 1, 1)), dim=2)
        moments = torch.cat(
            [self._encode_clip(pixels[:, :, index * self.clip_length : (index + 1) * self.clip_length]) for index in range(pixels.shape[2] // self.clip_length)],
            dim=2,
        )
        if self.token_drop > 0:
            moments = moments[:, :, : -self.token_drop]
        return moments

    @staticmethod
    def _sample_posterior(moments: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        # Diffusers' randn_tensor preserves a CPU generator by drawing on CPU
        # and moving the result, even when the posterior itself is on CUDA.
        noise = torch.randn(mean.shape, generator=generator, device="cpu", dtype=mean.dtype).to(mean.device)
        return mean + torch.exp(0.5 * logvar) * noise

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.latents_mean.to(latents.device).view(1, -1, 1, 1, 1)
        std = self.latents_std.to(latents.device).view(1, -1, 1, 1, 1)
        return (latents.float() - mean) / std

    def preprocess(self, pixels: torch.Tensor) -> torch.Tensor:
        mean = self.pixel_mean.to(pixels.device).view(1, -1, 1, 1, 1)
        std = self.pixel_std.to(pixels.device).view(1, -1, 1, 1, 1)
        return (pixels.float() - mean) / std

    def encode_condition(self, pixels: torch.Tensor, *, video: bool = False, return_cpu: bool = True) -> torch.Tensor:
        """Encode an RGB ``[1,3,F,H,W]`` reference with the released seed-42 posterior."""
        try:
            if pixels.ndim != 5 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
                raise ValueError(f"reference pixels must be [1,3,F,H,W], got {tuple(pixels.shape)}")
            device = self._activate()
            pixels = self.preprocess(pixels.to(device=device, dtype=torch.float32))
            with torch.no_grad():
                moments = self._encode(pixels) if video else self._encode_clip(pixels)
                generator = torch.Generator(device="cpu").manual_seed(42)
                latents = self._sample_posterior(moments, generator)
                latents = latents.to(torch.float16).float()
                latents = self.normalize_latents(latents)
            return latents.cpu() if return_cpu else latents
        finally:
            if self.cpu_offload:
                self.offload()

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        tokens_chunk_size = self.tokens_chunk_size
        temporal_ratio = self.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio

        num_tokens = latents.shape[2] + self.token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(self.token_drop > 0)
        if num_chunks <= 0:
            raise ValueError(f"Video latent sequence is too short for clip_length={self.clip_length}, token_drop={self.token_drop}: got {latents.shape[2]} frames")
        if pad_tokens > 0:
            latents = torch.cat(
                [latents, latents[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)],
                dim=2,
            )

        decoded_chunks: list[torch.Tensor] = []
        overlap = None
        for chunk_index in range(num_chunks):
            start = chunk_index * tokens_chunk_size
            clip = self._decode_clip(latents[:, :, start : start + tokens_chunk_size + self.token_overlap])
            for overlap_index in range(int(self.token_drop > 0) + 1):
                frame_start = overlap_index * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, self.frame_pre_padding :]
                if overlap_index == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                    decoded_chunks.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded_chunks.append(overlap)

        decoded = torch.cat(decoded_chunks, dim=2)
        if pad_tokens > 0:
            intra_tail = self.clip_length % temporal_ratio
            num_tokens_before_pad = latents.shape[2] - pad_tokens
            pad_frames = sum(intra_tail if intra_tail and (num_tokens_before_pad + offset) % tokens_chunk_size == 0 else temporal_ratio for offset in range(pad_tokens))
            decoded = decoded[:, :, :-pad_frames]
        return decoded

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.latents_mean.to(device=latents.device).view(1, -1, 1, 1, 1)
        std = self.latents_std.to(device=latents.device).view(1, -1, 1, 1, 1)
        return latents.float() * std + mean

    def postprocess(self, video: torch.Tensor) -> torch.Tensor:
        mean = self.pixel_mean.to(device=video.device).view(1, -1, 1, 1, 1)
        std = self.pixel_std.to(device=video.device).view(1, -1, 1, 1, 1)
        return (video.float() * std + mean).clamp_(0, 1)

    def _activate(self) -> torch.device:
        if self.cpu_offload:
            self.to(self.execution_device)
        return next(self.parameters()).device

    def offload(self) -> None:
        self.to("cpu")
        _empty_device_cache(self.execution_device)
        gc.collect()

    def _run_decode(
        self,
        latents: torch.Tensor,
        *,
        denormalize: bool,
        postprocess: bool,
        return_cpu: bool | None,
    ) -> torch.Tensor:
        try:
            if latents.ndim != 5 or latents.shape[1] != self.post_quant_conv.in_channels:
                raise ValueError(f"video latents must have shape [batch, {self.post_quant_conv.in_channels}, frames, height, width], got {tuple(latents.shape)}")
            device = self._activate()
            latents = latents.to(device=device, dtype=torch.float32)
            if denormalize:
                # Denormalization happens in FP32, before the released
                # FP16-autocast decoder recipe.
                latents = self.denormalize_latents(latents)
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with torch.no_grad(), autocast_context:
                video = self._decode(latents)
            video = video.float()
            if postprocess:
                video = self.postprocess(video)

            if return_cpu is None:
                return_cpu = self.cpu_offload
            if return_cpu:
                video = video.cpu()
            return video
        finally:
            if self.cpu_offload:
                self.offload()

    def decode_raw(self, denormalized_latents: torch.Tensor, *, return_cpu: bool | None = None) -> torch.Tensor:
        """Decode VAE-space latents to ImageNet-normalized RGB."""

        return self._run_decode(
            denormalized_latents,
            denormalize=False,
            postprocess=False,
            return_cpu=return_cpu,
        )

    def decode(self, latents: torch.Tensor, *, return_cpu: bool | None = None) -> torch.Tensor:
        """Decode normalized diffusion latents to RGB in ``[0, 1]``.

        Args:
            latents: ``[B, 24, T, H, W]`` diffusion-space video latents.
            return_cpu: Move the result to CPU. Defaults to ``cpu_offload``.
        """

        return self._run_decode(
            latents,
            denormalize=True,
            postprocess=True,
            return_cpu=return_cpu,
        )


# Explicit alias for callers that prefer the upstream autoencoder naming.
AutoencoderKLMiniMaxH3Native = MiniMaxH3VideoVAE
