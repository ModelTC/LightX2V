"""Native Qwen3-VL vision tower used by MiniMax-H3 conditioning."""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open


def _prepare_comfy_images(images):
    patches, grids = [], []
    patch_size, temporal_patch_size, merge_size = 16, 2, 2
    factor = patch_size * merge_size
    for image in images:
        pixels = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1)
        height, width = pixels.shape[-2:]
        h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
        if h_bar * w_bar > 12845056:
            beta = math.sqrt(height * width / 12845056)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < 3136:
            beta = math.sqrt(3136 / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        pixels = F.interpolate(pixels.unsqueeze(0), size=(h_bar, w_bar), mode="bilinear", align_corners=False).squeeze(0)
        pixels = (pixels - 0.5) / 0.5
        grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
        pixels = pixels.unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)
        pixels = pixels.reshape(1, temporal_patch_size, 3, grid_h // merge_size, merge_size, patch_size, grid_w // merge_size, merge_size, patch_size)
        pixels = pixels.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        patches.append(pixels.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size**2))
        grids.append((1, grid_h, grid_w))
    return {"pixel_values": torch.cat(patches), "image_grid_thw": torch.tensor(grids, dtype=torch.long)}


def _vision_position_ids(grid_thw, merge_size):
    positions = []
    device = grid_thw.device
    for t, h, w in grid_thw.tolist():
        hpos, wpos = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        shape = (h // merge_size, merge_size, w // merge_size, merge_size)
        hpos = hpos.reshape(shape).transpose(1, 2).flatten()
        wpos = wpos.reshape(shape).transpose(1, 2).flatten()
        positions.append(torch.stack((hpos, wpos), dim=-1).repeat(t, 1))
    return torch.cat(positions)


def _vision_cu_seqlens(grid_thw):
    values = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(0, dtype=torch.int32)
    return F.pad(values, (1, 0), value=0)


def _bilinear_indices_weights(grid_thw, side, merge_size):
    device = grid_thw.device
    index_parts = [[] for _ in range(4)]
    weight_parts = [[] for _ in range(4)]
    for t, h, w in grid_thw.tolist():
        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)
        h_floor, w_floor = h_grid.int(), w_grid.int()
        h_ceil, w_ceil = (h_floor + 1).clamp(max=side - 1), (w_floor + 1).clamp(max=side - 1)
        h_frac, w_frac = h_grid - h_floor, w_grid - w_floor
        corners = (
            (h_floor[:, None] * side + w_floor[None]).flatten(),
            (h_floor[:, None] * side + w_ceil[None]).flatten(),
            (h_ceil[:, None] * side + w_floor[None]).flatten(),
            (h_ceil[:, None] * side + w_ceil[None]).flatten(),
        )
        weights = (
            ((1 - h_frac)[:, None] * (1 - w_frac)[None]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None]).flatten(),
            (h_frac[:, None] * w_frac[None]).flatten(),
        )
        h_idx = torch.arange(h, device=device).view(h // merge_size, merge_size)
        w_idx = torch.arange(w, device=device).view(w // merge_size, merge_size)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None]).transpose(1, 2).flatten().repeat(t)
        for index in range(4):
            index_parts[index].append(corners[index][reorder])
            weight_parts[index].append(weights[index][reorder])
    return torch.stack([torch.cat(part) for part in index_parts]), torch.stack([torch.cat(part) for part in weight_parts])


def _rotate_half(value):
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_comfy_rope(value, cos, sin):
    rotated = value * cos
    midpoint = value.shape[-1] // 2
    rotated[..., :midpoint].addcmul_(value[..., midpoint:], -sin[..., midpoint:])
    rotated[..., midpoint:].addcmul_(value[..., :midpoint], sin[..., :midpoint])
    return rotated.to(value.dtype)


class _PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config["in_channels"]
        self.temporal_patch_size = config["temporal_patch_size"]
        self.patch_size = config["patch_size"]
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(self.in_channels, config["hidden_size"], kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, pixels):
        pixels = pixels.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(pixels.to(self.proj.weight.dtype)).view(-1, self.proj.out_channels)


class _VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config["num_heads"]
        self.head_dim = config["hidden_size"] // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.h3_implementation = config.get("h3_implementation")
        self.qkv = nn.Linear(config["hidden_size"], config["hidden_size"] * 3, bias=True)
        self.proj = nn.Linear(config["hidden_size"], config["hidden_size"], bias=True)

    def forward(self, hidden_states, cu_seqlens, cos, sin):
        length = hidden_states.shape[0]
        query, key, value = self.qkv(hidden_states).reshape(length, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        cos_f, sin_f = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        if self.h3_implementation == "comfyui":
            query = _apply_comfy_rope(query, cos_f, sin_f)
            key = _apply_comfy_rope(key, cos_f, sin_f)
        else:
            query_f, key_f = query.float(), key.float()
            query = (query_f * cos_f + _rotate_half(query_f) * sin_f).to(query.dtype)
            key = (key_f * cos_f + _rotate_half(key_f) * sin_f).to(key.dtype)
        outputs = []
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            q = query[start:end].transpose(0, 1).unsqueeze(0)
            k = key[start:end].transpose(0, 1).unsqueeze(0)
            v = value[start:end].transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
            outputs.append(out.transpose(1, 2).reshape(end - start, -1))
        return self.proj(torch.cat(outputs, dim=0))


class _VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=True)
        self.linear_fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=True)

    def forward(self, hidden_states):
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_states), approximate="tanh"))


class _VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config["hidden_size"], eps=1e-6)
        self.norm2 = nn.LayerNorm(config["hidden_size"], eps=1e-6)
        self.attn = _VisionAttention(config)
        self.mlp = _VisionMLP(config)

    def forward(self, hidden_states, cu_seqlens, cos, sin):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, cos, sin)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class _PatchMerger(nn.Module):
    def __init__(self, config, postshuffle=False):
        super().__init__()
        merged_size = config["hidden_size"] * config["spatial_merge_size"] ** 2
        self.merged_size = merged_size
        self.postshuffle = postshuffle
        self.norm = nn.LayerNorm(merged_size if postshuffle else config["hidden_size"], eps=1e-6)
        self.linear_fc1 = nn.Linear(merged_size, merged_size)
        self.linear_fc2 = nn.Linear(merged_size, config["out_hidden_size"])

    def forward(self, hidden_states):
        if self.postshuffle:
            hidden_states = self.norm(hidden_states.view(-1, self.merged_size))
        else:
            hidden_states = self.norm(hidden_states).view(-1, self.merged_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_states)))


class MiniMaxH3Qwen3VLVisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.spatial_merge_size = int(config["spatial_merge_size"])
        self.patch_embed = _PatchEmbed(config)
        self.pos_embed = nn.Embedding(config["num_position_embeddings"], config["hidden_size"])
        self.blocks = nn.ModuleList([_VisionBlock(config) for _ in range(config["depth"])])
        self.merger = _PatchMerger(config)
        self.deepstack_visual_indexes = list(config["deepstack_visual_indexes"])
        self.deepstack_merger_list = nn.ModuleList([_PatchMerger(config, postshuffle=True) for _ in self.deepstack_visual_indexes])
        head_dim = config["hidden_size"] // config["num_heads"]
        self.register_buffer("rotary_inv_freq", 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2))), persistent=False)

    def forward(self, pixels, grid_thw):
        grid_thw = grid_thw.to(device=pixels.device)
        indices, weights = _bilinear_indices_weights(grid_thw, int(math.sqrt(self.config["num_position_embeddings"])), self.spatial_merge_size)
        position_ids = _vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = _vision_cu_seqlens(grid_thw)
        hidden_states = self.patch_embed(pixels)
        if self.config.get("h3_implementation") == "comfyui":
            # Comfy keeps the position table and interpolation arithmetic in the checkpoint dtype.
            corners = self.pos_embed(indices) * weights.to(self.pos_embed.weight.dtype)[:, :, None]
            pos_embed = corners[0] + corners[1] + corners[2] + corners[3]
        else:
            pos_embed = (self.pos_embed(indices) * weights[:, :, None]).sum(0)
        hidden_states = hidden_states + pos_embed.to(hidden_states.dtype)
        rotary = (position_ids.unsqueeze(-1) * self.rotary_inv_freq.to(position_ids.device)).flatten(1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        cos, sin = rotary.cos(), rotary.sin()
        deepstack = []
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, cos, sin)
            if layer_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack.append(self.deepstack_merger_list[merger_index](hidden_states))
        return self.merger(hidden_states), deepstack

    @classmethod
    def from_pretrained(cls, text_encoder_path, vision_config):
        root = Path(text_encoder_path)
        with torch.device("meta"):
            model = cls(vision_config)
        with (root / "model.safetensors.index.json").open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        prefix = "model.visual."
        names = {name: shard for name, shard in weight_map.items() if name.startswith(prefix)}
        by_shard = defaultdict(list)
        for name, shard in names.items():
            by_shard[shard].append(name)
        state = {}
        for shard, shard_names in by_shard.items():
            logger.info("Loading native Qwen3-VL vision tensors from {}", shard)
            with safe_open(root / shard, framework="pt", device="cpu") as checkpoint:
                for name in shard_names:
                    state[name[len(prefix) :]] = checkpoint.get_tensor(name)
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        # rotary_inv_freq is non-persistent, so every persistent tensor must match.
        if missing or unexpected:
            raise RuntimeError(f"Qwen3-VL vision checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        head_dim = vision_config["hidden_size"] // vision_config["num_heads"]
        model.rotary_inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float32) / (head_dim // 2)))
        if vision_config.get("h3_implementation") == "comfyui":
            position_dtype = model.pos_embed.weight.dtype
            model.float()
            model.pos_embed.to(dtype=position_dtype)
        return model.eval().requires_grad_(False)


__all__ = ["MiniMaxH3Qwen3VLVisionTower"]
