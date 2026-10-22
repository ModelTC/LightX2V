# SPDX-License-Identifier: Apache-2.0
"""Native StreamRender-H3 24-layer causal ViT video decoder.

The released decoder is a two-latent stream with four-latent K/V history,
shifted local attention windows, and one global register set per chunk.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import MiniMaxH3VideoRotaryPosEmbed

_PATCH = 16
_PATCH_T = 4


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_qkv = nn.Linear(2048, 6144)
        self.to_out = nn.Linear(2048, 2048)
        self.norm_q = nn.RMSNorm(64, eps=1e-5, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(64, eps=1e-5, elementwise_affine=False)


class _FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(2048, 16384)
        self.w2 = nn.Linear(8192, 2048)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, value = self.w1(hidden).chunk(2, dim=-1)
        return self.w2(F.silu(gate).mul_(value))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(2048, eps=1e-5)
        self.attn = _Attention()
        self.scale1 = nn.Parameter(torch.empty(2048))
        self.norm2 = nn.RMSNorm(2048, eps=1e-5)
        self.ff = _FeedForward()
        self.scale2 = nn.Parameter(torch.empty(2048))


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(tensor.dtype)
    sin = sin.to(tensor.dtype)
    rotary_dim = cos.shape[-1]
    if rotary_dim < tensor.shape[-1]:
        rotary, passthrough = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
        scaled = rotary * cos
        scaled.add_(_rotate_half(rotary) * sin)
        return torch.cat((scaled, passthrough), dim=-1)
    scaled = tensor * cos
    scaled.add_(_rotate_half(tensor) * sin)
    return scaled


class StreamRenderH3CausalDecoder(nn.Module):
    """Checkpoint-compatible eager decoder; no upstream runtime imports."""

    def __init__(self, checkpoint: str | Path, config: dict, device: str | torch.device = "cuda"):
        super().__init__()
        self.window = 16
        self.chunk = 2
        self.cache_latents = 4
        self.block_latents = 5
        self.frames_to_trim = 3
        self.time_norm = 7.0
        self.heads = 32
        self.head_dim = 64
        with torch.device("meta"):
            self.post_quant_conv = nn.Conv3d(24, 24, 1)
            self.x_embedder = nn.Linear(24, 2048)
            self.register_tokens = nn.Parameter(torch.empty(1, 4, 2048))
            self.blocks = nn.ModuleList(_Block() for _ in range(24))
            self.norm_out = nn.LayerNorm(2048, eps=1e-5)
            self.proj_out = nn.Linear(2048, 3072)
        self.pos_embed = MiniMaxH3VideoRotaryPosEmbed(48, theta=100)
        self.register_buffer("latents_mean", torch.tensor(config["latents_mean"], dtype=torch.float32), persistent=False)
        self.register_buffer("latents_std", torch.tensor(config["latents_std"], dtype=torch.float32), persistent=False)
        self.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True, assign=True)
        self.to(device).eval()

    @classmethod
    def from_model_path(cls, model_path: str | Path, device: str | torch.device = "cuda"):
        root = Path(model_path)
        runtime = json.loads((root / "configs/runtime.json").read_text())
        args = runtime["inference"]["models"]["video_decoder"]["args"]
        assets = json.loads((root / "assets.json").read_text())["assets"]
        checkpoint = root / assets["causal_decoder"]
        return cls(checkpoint, args["decoder_config"], device)

    def create_stream(self):
        return StreamRenderH3DecodeStream(self)

    def _ids(self, start: int, count: int, height: int, width: int, shift: int, device: torch.device):
        temporal = (torch.arange(start, start + count, device=device, dtype=torch.float32) + 0.5) / self.time_norm * 2 - 1
        yy = ((torch.arange(height, device=device) + shift) % self.window).float()
        xx = ((torch.arange(width, device=device) + shift) % self.window).float()
        yy = (yy + 0.5) / self.window * 2 - 1
        xx = (xx + 0.5) / self.window * 2 - 1
        grid = torch.stack(torch.meshgrid(temporal, yy, xx, indexing="ij"), -1).reshape(1, -1, 3)
        return self.pos_embed(grid)

    def _suffix_ids(self, start: int, count: int, device: torch.device):
        grid = torch.zeros((1, count, 3), device=device, dtype=torch.float32)
        grid[..., 0] = (start + 0.5) / self.time_norm * 2 - 1
        return self.pos_embed(grid)

    def _partition(self, values: torch.Tensor, shift: int):
        batch, frames, height, width = values.shape[:4]
        padded_height = ((height + shift + self.window - 1) // self.window) * self.window
        padded_width = ((width + shift + self.window - 1) // self.window) * self.window
        tail = values.shape[4:]
        padded = values.new_zeros(batch, frames, padded_height, padded_width, *tail)
        padded[:, :, shift : shift + height, shift : shift + width] = values
        valid = torch.zeros(padded_height, padded_width, dtype=torch.bool, device=values.device)
        valid[shift : shift + height, shift : shift + width] = True
        windows_h, windows_w = padded_height // self.window, padded_width // self.window
        windows = padded.view(batch, frames, windows_h, self.window, windows_w, self.window, *tail)
        windows = windows.permute(0, 2, 4, 1, 3, 5, *range(6, 6 + len(tail)))
        windows = windows.reshape(batch * windows_h * windows_w, frames * self.window**2, *tail)
        valid = valid.view(windows_h, self.window, windows_w, self.window).permute(0, 2, 1, 3)
        valid = valid.reshape(windows_h * windows_w, 1, self.window**2).expand(-1, frames, -1)
        return windows, valid.reshape(windows_h * windows_w, frames * self.window**2), (batch, frames, height, width, padded_height, padded_width, windows_h, windows_w)

    def _unpartition(self, windows: torch.Tensor, shape: tuple[int, ...], shift: int) -> torch.Tensor:
        batch, frames, height, width, padded_height, padded_width, windows_h, windows_w = shape
        tail = windows.shape[2:]
        values = windows.view(batch, windows_h, windows_w, frames, self.window, self.window, *tail)
        values = values.permute(0, 3, 1, 4, 2, 5, *range(6, 6 + len(tail)))
        values = values.reshape(batch, frames, padded_height, padded_width, *tail)
        return values[:, :, shift : shift + height, shift : shift + width]

    @torch.no_grad()
    def decode_chunk(self, latents: torch.Tensor, state: dict) -> torch.Tensor:
        """Decode one NTCHW pair into ImageNet-normalized NCTHW RGB."""
        batch, count, _, height, width = latents.shape
        device = latents.device
        compute_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else latents.dtype
        raw = latents.float() * self.latents_std.view(1, 1, -1, 1, 1) + self.latents_mean.view(1, 1, -1, 1, 1)
        raw = self.post_quant_conv(raw.transpose(1, 2))
        with torch.autocast("cuda", enabled=False):
            hidden = self.x_embedder(raw.permute(0, 2, 3, 4, 1).float())
        hidden = hidden.to(compute_dtype)
        suffix_count = self.register_tokens.shape[1] + 1
        suffix = torch.cat((self.register_tokens.expand(batch, -1, -1), hidden.new_zeros(batch, 1, 2048)), 1).to(compute_dtype)
        start = state["t"]
        for layer_id, block in enumerate(self.blocks):
            shift = self.window // 2 if layer_id % 2 else 0
            norm = block.norm1(hidden.float()).to(hidden.dtype)
            norm_suffix = block.norm1(suffix.float()).to(suffix.dtype)
            query, key, value = block.attn.to_qkv(norm).view(batch, count, height, width, self.heads, 3 * self.head_dim).chunk(3, dim=-1)
            suffix_query, suffix_key, suffix_value = block.attn.to_qkv(norm_suffix).view(batch, suffix_count, self.heads, 3 * self.head_dim).chunk(3, dim=-1)
            query = block.attn.norm_q(query.float()).to(query.dtype)
            suffix_query = block.attn.norm_q(suffix_query.float()).to(suffix_query.dtype)
            key = block.attn.norm_k(key.float()).to(key.dtype)
            suffix_key = block.attn.norm_k(suffix_key.float()).to(suffix_key.dtype)
            cos, sin = self._ids(start, count, height, width, shift, device)
            query = _apply_rotary(query.reshape(batch, count * height * width, self.heads, self.head_dim), cos, sin).view(batch, count, height, width, self.heads, self.head_dim)
            key = _apply_rotary(key.reshape(batch, count * height * width, self.heads, self.head_dim), cos, sin).view(batch, count, height, width, self.heads, self.head_dim)
            suffix_cos, suffix_sin = self._suffix_ids(start, suffix_count, device)
            suffix_query = _apply_rotary(suffix_query, suffix_cos, suffix_sin)
            suffix_key = _apply_rotary(suffix_key, suffix_cos, suffix_sin)
            keys = key if state["keys"][layer_id] is None else torch.cat((state["keys"][layer_id], key), dim=1)
            values = value if state["values"][layer_id] is None else torch.cat((state["values"][layer_id], value), dim=1)
            key_frames = keys.shape[1]
            window_queries, _, shape = self._partition(query, shift)
            window_keys, valid, _ = self._partition(keys, shift)
            window_values, _, _ = self._partition(values, shift)
            windows_per_sample = window_queries.shape[0] // batch
            suffix_keys = (
                suffix_key.unsqueeze(1).expand(batch, windows_per_sample, suffix_count, self.heads, self.head_dim).reshape(batch * windows_per_sample, suffix_count, self.heads, self.head_dim)
            )
            suffix_values = (
                suffix_value.unsqueeze(1).expand(batch, windows_per_sample, suffix_count, self.heads, self.head_dim).reshape(batch * windows_per_sample, suffix_count, self.heads, self.head_dim)
            )
            all_keys = torch.cat((window_keys, suffix_keys), dim=1).transpose(1, 2)
            all_values = torch.cat((window_values, suffix_values), dim=1).transpose(1, 2)
            mask = torch.cat((valid, torch.ones(windows_per_sample, suffix_count, dtype=torch.bool, device=device)), dim=1)
            mask = mask.repeat(batch, 1)[:, None, None, :]
            window_output = F.scaled_dot_product_attention(window_queries.transpose(1, 2), all_keys, all_values, attn_mask=mask).transpose(1, 2)
            output = self._unpartition(window_output, shape, shift).reshape(batch, count, height, width, 2048)
            all_keys = torch.cat((keys.reshape(batch, key_frames * height * width, self.heads, self.head_dim), suffix_key), dim=1).transpose(1, 2)
            all_values = torch.cat((values.reshape(batch, key_frames * height * width, self.heads, self.head_dim), suffix_value), dim=1).transpose(1, 2)
            suffix_output = F.scaled_dot_product_attention(suffix_query.transpose(1, 2), all_keys, all_values).transpose(1, 2).reshape(batch, suffix_count, 2048)
            hidden = hidden + block.attn.to_out(output) * block.scale1
            suffix = suffix + block.attn.to_out(suffix_output) * block.scale1
            hidden = hidden + block.ff(block.norm2(hidden.float()).to(hidden.dtype)) * block.scale2
            suffix = suffix + block.ff(block.norm2(suffix.float()).to(suffix.dtype)) * block.scale2
            state["keys"][layer_id] = keys[:, -self.cache_latents :]
            state["values"][layer_id] = values[:, -self.cache_latents :]
        hidden = self.norm_out(hidden)
        with torch.autocast("cuda", enabled=False):
            pixels = self.proj_out(hidden.float())
        pixels = pixels.view(batch, count, height, width, 3, _PATCH_T, _PATCH, _PATCH)
        pixels = pixels.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(batch, 3, count * _PATCH_T, height * _PATCH, width * _PATCH)
        keep = torch.ones(count * _PATCH_T, dtype=torch.bool, device=device)
        for index in range(count):
            if (start + index) % self.block_latents == 0:
                keep[index * _PATCH_T : index * _PATCH_T + self.frames_to_trim] = False
        state["t"] = start + count
        return pixels[:, :, keep]


class StreamRenderH3DecodeStream:
    """Two-latent causal output stream sharing the decoder's immutable weights."""

    def __init__(self, decoder: StreamRenderH3CausalDecoder):
        self.decoder = decoder
        self.state = {"t": 0, "keys": [None] * 24, "values": [None] * 24}
        self.pending = []

    @staticmethod
    def _rgb(pixels: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor((0.485, 0.456, 0.406), device=pixels.device).view(1, 3, 1, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=pixels.device).view(1, 3, 1, 1, 1)
        return pixels.float().mul_(std).add_(mean).clamp_(0, 1).transpose(1, 2).contiguous()

    @torch.no_grad()
    def push(self, latents: torch.Tensor) -> torch.Tensor | None:
        """Consume normalized NTCHW H3 latents and publish complete RGB pairs."""
        self.pending.extend(latents[:, index : index + 1] for index in range(latents.shape[1]))
        outputs = []
        while len(self.pending) >= 2:
            pair = torch.cat(self.pending[:2], dim=1)
            del self.pending[:2]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pixels = self.decoder.decode_chunk(pair, self.state)
            outputs.append(self._rgb(pixels))
        return torch.cat(outputs, dim=1) if outputs else None

    @torch.no_grad()
    def flush(self) -> torch.Tensor | None:
        if not self.pending:
            return None
        latent = self.pending.pop()
        self.pending.clear()
        paired = torch.cat((latent, latent), dim=1)
        start = self.state["t"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pixels = self.decoder.decode_chunk(paired, self.state)
        own_frames = _PATCH_T - (self.decoder.frames_to_trim if start % self.decoder.block_latents == 0 else 0)
        return self._rgb(pixels[:, :, :own_frames])


__all__ = ["StreamRenderH3CausalDecoder", "StreamRenderH3DecodeStream"]
