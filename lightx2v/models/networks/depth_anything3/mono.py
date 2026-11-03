"""Depth Anything 3 monocular network for the ComfyUI-format checkpoint.

The architecture follows the Apache-2.0 Depth Anything 3 implementation by
ByteDance Seed and its Apache-2.0 DINOv2 backbone by Meta. This module keeps
only the single-view ViT-L + DPT/sky path; camera and multi-view heads are not
part of the mono-large checkpoint.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbeddings(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Conv2d(3, hidden_size, kernel_size=14, stride=14)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(image).flatten(2).transpose(1, 2)


class Embeddings(nn.Module):
    def __init__(self, hidden_size: int, image_size: int = 518):
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        patches = (image_size // 14) ** 2
        self.position_embeddings = nn.Parameter(torch.empty(1, patches + 1, hidden_size))
        self.patch_embeddings = PatchEmbeddings(hidden_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = image.shape
        patches = self.patch_embeddings(image)
        tokens = torch.cat((self.cls_token.expand(batch, -1, -1), patches), dim=1)
        patch_h, patch_w = height // 14, width // 14
        pos = self.position_embeddings
        original_side = math.isqrt(pos.shape[1] - 1)
        if patch_h != original_side or patch_w != original_side:
            patch_pos = pos[:, 1:].float().reshape(1, original_side, original_side, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(
                patch_pos,
                scale_factor=((patch_h + 0.1) / original_side, (patch_w + 0.1) / original_side),
                mode="bicubic",
                antialias=False,
            )
            if patch_pos.shape[-2:] != (patch_h, patch_w):
                raise ValueError(f"DA3 positional embedding size mismatch: {patch_pos.shape[-2:]} vs {(patch_h, patch_w)}")
            pos = torch.cat((pos[:, :1].float(), patch_pos.flatten(2).transpose(1, 2)), dim=1).to(tokens.dtype)
        return tokens + pos


class AttentionCore(nn.Module):
    def __init__(self, hidden_size: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden_size = x.shape
        head_size = hidden_size // self.heads

        def split(linear: nn.Linear) -> torch.Tensor:
            return linear(x).reshape(batch, tokens, self.heads, head_size).transpose(1, 2)

        result = F.scaled_dot_product_attention(split(self.query), split(self.key), split(self.value))
        return result.transpose(1, 2).reshape(batch, tokens, hidden_size)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int):
        super().__init__()
        self.attention = AttentionCore(hidden_size, heads)
        self.output = nn.Module()
        self.output.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output.dense(self.attention(x))


class LayerScale(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.empty(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.lambda1


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attention = Attention(hidden_size, heads)
        self.layer_scale1 = LayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(hidden_size)
        self.layer_scale2 = LayerScale(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class Encoder(nn.Module):
    def __init__(self, hidden_size: int, heads: int, depth: int):
        super().__init__()
        self.layer = nn.ModuleList(EncoderLayer(hidden_size, heads) for _ in range(depth))


class MonoBackbone(nn.Module):
    def __init__(self, hidden_size: int = 1024, heads: int = 16, depth: int = 24, out_layers: tuple[int, ...] = (4, 11, 17, 23)):
        super().__init__()
        self.embeddings = Embeddings(hidden_size)
        self.encoder = Encoder(hidden_size, heads, depth)
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.out_layers = out_layers

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        x = self.embeddings(image)
        output = []
        for index, block in enumerate(self.encoder.layer):
            x = block(x)
            if index in self.out_layers:
                output.append(self.layernorm(x)[:, 1:])
        return output


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(x))
        y = self.conv2(F.relu(y))
        return x + y


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, has_residual: bool = True):
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(features) if has_residual else None
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1)

    def forward(self, top: torch.Tensor, side: torch.Tensor | None = None, *, size: tuple[int, int] | None = None) -> torch.Tensor:
        if side is not None and self.resConfUnit1 is not None:
            top = top + self.resConfUnit1(side)
        top = self.resConfUnit2(top)
        if size is None:
            top = F.interpolate(top, scale_factor=2, mode="bilinear", align_corners=True)
        else:
            top = F.interpolate(top, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(top)


class DPTHead(nn.Module):
    def __init__(self, hidden_size: int = 1024, features: int = 256, out_channels: tuple[int, ...] = (256, 512, 1024, 1024)):
        super().__init__()
        self.projects = nn.ModuleList(nn.Conv2d(hidden_size, channels, 1) for channels in out_channels)
        self.resize_layers = nn.ModuleList(
            (
                nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
            )
        )
        self.scratch = nn.Module()
        for index, channels in enumerate(out_channels, 1):
            setattr(self.scratch, f"layer{index}_rn", nn.Conv2d(channels, features, 3, padding=1, bias=False))
        for index in range(1, 5):
            setattr(self.scratch, f"refinenet{index}", FeatureFusionBlock(features, has_residual=index != 4))
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, 3, padding=1)
        self.scratch.output_conv2 = nn.Sequential(nn.Conv2d(features // 2, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1))
        self.scratch.sky_output_conv2 = nn.Sequential(nn.Conv2d(features // 2, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1))

    def forward(self, features: list[torch.Tensor], height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch_h, patch_w = height // 14, width // 14
        resized = []
        for index, feature in enumerate(features):
            image_feature = feature.transpose(1, 2).reshape(feature.shape[0], feature.shape[2], patch_h, patch_w)
            resized.append(self.resize_layers[index](self.projects[index](image_feature)))
        scratch = self.scratch
        l1 = scratch.layer1_rn(resized[0])
        l2 = scratch.layer2_rn(resized[1])
        l3 = scratch.layer3_rn(resized[2])
        l4 = scratch.layer4_rn(resized[3])
        fused = scratch.refinenet4(l4, size=l3.shape[-2:])
        fused = scratch.refinenet3(fused, l3, size=l2.shape[-2:])
        fused = scratch.refinenet2(fused, l2, size=l1.shape[-2:])
        fused = scratch.refinenet1(fused, l1)
        fused = scratch.output_conv1(fused)
        fused = F.interpolate(fused, size=(height, width), mode="bilinear", align_corners=True)
        depth = torch.exp(scratch.output_conv2(fused)).squeeze(1)
        sky = F.relu(scratch.sky_output_conv2(fused)).squeeze(1)
        return depth, sky


class DepthAnything3Mono(nn.Module):
    """DA3MONO-LARGE with checkpoint names matching the provided safetensors."""

    def __init__(self):
        super().__init__()
        self.backbone = MonoBackbone()
        self.head = DPTHead()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image.shape[-2:]
        if height % 14 or width % 14:
            raise ValueError("Depth Anything 3 input dimensions must be divisible by 14")
        with torch.autocast(device_type=image.device.type, dtype=torch.bfloat16, enabled=image.device.type == "cuda"):
            features = self.backbone(image)
        return self.head([feature.float() for feature in features], height, width)
