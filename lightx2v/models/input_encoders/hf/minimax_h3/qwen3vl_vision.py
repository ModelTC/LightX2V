"""Native Qwen3-VL vision tower used by MiniMax-H3 conditioning."""

import gc
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open


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


def _replace_parameter(module, name, value):
    setattr(module, name, nn.Parameter(value.contiguous(), requires_grad=False))


def _column_shard(value, tp_rank, tp_size):
    if value.shape[0] % tp_size:
        raise ValueError(f"Cannot column-shard shape {tuple(value.shape)} over vision TP size {tp_size}")
    shard_size = value.shape[0] // tp_size
    return value.narrow(0, tp_rank * shard_size, shard_size)


def _row_shard(value, tp_rank, tp_size):
    if value.shape[1] % tp_size:
        raise ValueError(f"Cannot row-shard shape {tuple(value.shape)} over vision TP size {tp_size}")
    shard_size = value.shape[1] // tp_size
    return value.narrow(1, tp_rank * shard_size, shard_size)


def _row_parallel_linear(module, hidden_states, tp_group, tp_rank, tp_size):
    if tp_size == 1:
        return module(hidden_states)
    # SGLang adds row-parallel bias on rank 0 before the reduction.
    bias = module.bias if tp_rank == 0 else None
    output = F.linear(hidden_states, module.weight, bias)
    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)
    return output


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
    def __init__(self, config, tp_group=None, tp_rank=0, tp_size=1):
        super().__init__()
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.total_num_heads = config["num_heads"]
        if self.total_num_heads % tp_size:
            raise ValueError(f"Qwen3-VL vision heads ({self.total_num_heads}) must be divisible by TP size ({tp_size})")
        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = config["hidden_size"] // self.total_num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(config["hidden_size"], config["hidden_size"] * 3, bias=True)
        self.proj = nn.Linear(config["hidden_size"], config["hidden_size"], bias=True)

    def shard_for_tensor_parallel(self):
        if self.tp_size == 1:
            return
        hidden_size = self.qkv.weight.shape[1]
        if hidden_size % self.tp_size:
            raise ValueError(f"Qwen3-VL vision hidden size ({hidden_size}) must be divisible by TP size ({self.tp_size})")
        local_size = hidden_size // self.tp_size
        qkv_weight = torch.cat(
            [self.qkv.weight.narrow(0, component * hidden_size + self.tp_rank * local_size, local_size) for component in range(3)],
            dim=0,
        )
        qkv_bias = torch.cat(
            [self.qkv.bias.narrow(0, component * hidden_size + self.tp_rank * local_size, local_size) for component in range(3)],
            dim=0,
        )
        _replace_parameter(self.qkv, "weight", qkv_weight)
        _replace_parameter(self.qkv, "bias", qkv_bias)
        _replace_parameter(self.proj, "weight", _row_shard(self.proj.weight, self.tp_rank, self.tp_size))

    def forward(self, hidden_states, cu_seqlens, cos, sin):
        length = hidden_states.shape[0]
        query, key, value = self.qkv(hidden_states).reshape(length, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        query_f, key_f = query.float(), key.float()
        cos_f, sin_f = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
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
        return _row_parallel_linear(self.proj, torch.cat(outputs, dim=0), self.tp_group, self.tp_rank, self.tp_size)


class _VisionMLP(nn.Module):
    def __init__(self, config, tp_group=None, tp_rank=0, tp_size=1):
        super().__init__()
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.linear_fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=True)
        self.linear_fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=True)

    def shard_for_tensor_parallel(self):
        if self.tp_size == 1:
            return
        _replace_parameter(self.linear_fc1, "weight", _column_shard(self.linear_fc1.weight, self.tp_rank, self.tp_size))
        _replace_parameter(self.linear_fc1, "bias", _column_shard(self.linear_fc1.bias, self.tp_rank, self.tp_size))
        _replace_parameter(self.linear_fc2, "weight", _row_shard(self.linear_fc2.weight, self.tp_rank, self.tp_size))

    def forward(self, hidden_states):
        hidden_states = F.gelu(self.linear_fc1(hidden_states), approximate="tanh")
        return _row_parallel_linear(self.linear_fc2, hidden_states, self.tp_group, self.tp_rank, self.tp_size)


class _VisionBlock(nn.Module):
    def __init__(self, config, tp_group=None, tp_rank=0, tp_size=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(config["hidden_size"], eps=1e-6)
        self.norm2 = nn.LayerNorm(config["hidden_size"], eps=1e-6)
        self.attn = _VisionAttention(config, tp_group, tp_rank, tp_size)
        self.mlp = _VisionMLP(config, tp_group, tp_rank, tp_size)

    def shard_for_tensor_parallel(self):
        self.attn.shard_for_tensor_parallel()
        self.mlp.shard_for_tensor_parallel()

    def forward(self, hidden_states, cu_seqlens, cos, sin):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, cos, sin)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class _PatchMerger(nn.Module):
    def __init__(self, config, postshuffle=False, tp_group=None, tp_rank=0, tp_size=1):
        super().__init__()
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        merged_size = config["hidden_size"] * config["spatial_merge_size"] ** 2
        self.merged_size = merged_size
        self.postshuffle = postshuffle
        self.norm = nn.LayerNorm(merged_size if postshuffle else config["hidden_size"], eps=1e-6)
        self.linear_fc1 = nn.Linear(merged_size, merged_size)
        self.linear_fc2 = nn.Linear(merged_size, config["out_hidden_size"])

    def shard_for_tensor_parallel(self):
        if self.tp_size == 1:
            return
        _replace_parameter(self.linear_fc1, "weight", _column_shard(self.linear_fc1.weight, self.tp_rank, self.tp_size))
        _replace_parameter(self.linear_fc1, "bias", _column_shard(self.linear_fc1.bias, self.tp_rank, self.tp_size))
        _replace_parameter(self.linear_fc2, "weight", _row_shard(self.linear_fc2.weight, self.tp_rank, self.tp_size))

    def forward(self, hidden_states):
        if self.postshuffle:
            hidden_states = self.norm(hidden_states.view(-1, self.merged_size))
        else:
            hidden_states = self.norm(hidden_states).view(-1, self.merged_size)
        hidden_states = F.gelu(self.linear_fc1(hidden_states))
        return _row_parallel_linear(self.linear_fc2, hidden_states, self.tp_group, self.tp_rank, self.tp_size)


class MiniMaxH3Qwen3VLVisionTower(nn.Module):
    def __init__(self, config, tp_group=None):
        super().__init__()
        self.config = dict(config)
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        self.spatial_merge_size = int(config["spatial_merge_size"])
        self.patch_embed = _PatchEmbed(config)
        self.pos_embed = nn.Embedding(config["num_position_embeddings"], config["hidden_size"])
        self.blocks = nn.ModuleList([_VisionBlock(config, tp_group, self.tp_rank, self.tp_size) for _ in range(config["depth"])])
        self.merger = _PatchMerger(config, tp_group=tp_group, tp_rank=self.tp_rank, tp_size=self.tp_size)
        self.deepstack_visual_indexes = list(config["deepstack_visual_indexes"])
        self.deepstack_merger_list = nn.ModuleList([_PatchMerger(config, postshuffle=True, tp_group=tp_group, tp_rank=self.tp_rank, tp_size=self.tp_size) for _ in self.deepstack_visual_indexes])
        head_dim = config["hidden_size"] // config["num_heads"]
        self.register_buffer("rotary_inv_freq", 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2))), persistent=False)

    def shard_for_tensor_parallel(self):
        if self.tp_size == 1:
            return self
        for block in self.blocks:
            block.shard_for_tensor_parallel()
        self.merger.shard_for_tensor_parallel()
        for merger in self.deepstack_merger_list:
            merger.shard_for_tensor_parallel()
        return self

    def forward(self, pixels, grid_thw):
        grid_thw = grid_thw.to(device=pixels.device)
        indices, weights = _bilinear_indices_weights(grid_thw, int(math.sqrt(self.config["num_position_embeddings"])), self.spatial_merge_size)
        position_ids = _vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = _vision_cu_seqlens(grid_thw)
        hidden_states = self.patch_embed(pixels)
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
    def from_pretrained(cls, text_encoder_path, vision_config, tp_group=None):
        root = Path(text_encoder_path)
        with torch.device("meta"):
            model = cls(vision_config, tp_group=tp_group)
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
        model.shard_for_tensor_parallel()
        state.clear()
        gc.collect()
        head_dim = vision_config["hidden_size"] // vision_config["num_heads"]
        model.rotary_inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float32) / (head_dim // 2)))
        if model.tp_size > 1:
            logger.info("Sharded native Qwen3-VL vision tower over rank {}/{}", model.tp_rank, model.tp_size)
        return model.eval().requires_grad_(False)


__all__ = ["MiniMaxH3Qwen3VLVisionTower"]
