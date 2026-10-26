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
    """Checkpoint-compatible causal decoder; no upstream runtime imports."""

    def __init__(self, checkpoint: str | Path, config: dict, device: str | torch.device = "cuda", *, linear_dtype: torch.dtype | None = None, use_compile: bool = False):
        super().__init__()
        self.use_compile = use_compile
        self._compiled_projection = None
        self._compiled_finish = None
        self._window_layouts = {}
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
        if linear_dtype is not None:
            if linear_dtype != torch.bfloat16:
                raise ValueError("The StreamRender BF16 stream requires BF16 block linear weights")
            # The stream already executes these projections in BF16 autocast.
            # Store that same representation once, avoiding an unused FP32
            # copy and repeated autocast copies for every two-latent decode.
            # Norms, input embedding and output projection retain FP32 weights.
            for module in self.blocks.modules():
                if isinstance(module, nn.Linear):
                    module.to(dtype=linear_dtype)
        self.to(device).eval()

    @classmethod
    def from_model_path(cls, model_path: str | Path, device: str | torch.device = "cuda", *, linear_dtype: torch.dtype | None = None, use_compile: bool = False):
        root = Path(model_path)
        runtime = json.loads((root / "configs/runtime.json").read_text())
        args = runtime["inference"]["models"]["video_decoder"]["args"]
        assets = json.loads((root / "assets.json").read_text())["assets"]
        checkpoint = root / assets["causal_decoder"]
        return cls(checkpoint, args["decoder_config"], device, linear_dtype=linear_dtype, use_compile=use_compile)

    def create_stream(self):
        return StreamRenderH3DecodeStream(self)

    def _project_block(self, block, hidden, suffix):
        batch, count, height, width = hidden.shape[:4]
        norm = block.norm1(hidden.float()).to(hidden.dtype)
        norm_suffix = block.norm1(suffix.float()).to(suffix.dtype)
        query, key, value = block.attn.to_qkv(norm).view(batch, count, height, width, self.heads, 3 * self.head_dim).chunk(3, dim=-1)
        suffix_query, suffix_key, suffix_value = block.attn.to_qkv(norm_suffix).view(batch, suffix.shape[1], self.heads, 3 * self.head_dim).chunk(3, dim=-1)
        query = block.attn.norm_q(query.float()).to(query.dtype)
        suffix_query = block.attn.norm_q(suffix_query.float()).to(suffix_query.dtype)
        key = block.attn.norm_k(key.float()).to(key.dtype)
        suffix_key = block.attn.norm_k(suffix_key.float()).to(suffix_key.dtype)
        return query, key, value, suffix_query, suffix_key, suffix_value

    @staticmethod
    def _finish_block(block, hidden, suffix, output, suffix_output):
        hidden = hidden + block.attn.to_out(output) * block.scale1
        suffix = suffix + block.attn.to_out(suffix_output) * block.scale1
        hidden = hidden + block.ff(block.norm2(hidden.float()).to(hidden.dtype)) * block.scale2
        suffix = suffix + block.ff(block.norm2(suffix.float()).to(suffix.dtype)) * block.scale2
        return hidden, suffix

    def _block_functions(self):
        if not self.use_compile:
            return self._project_block, self._finish_block
        if self._compiled_projection is None:
            # Mutable KV, window visibility, trimming and stream state remain
            # outside these pure arithmetic regions. Preserve BF16 cast points.
            options = {"emulate_precision_casts": True}
            self._compiled_projection = torch.compile(self._project_block, dynamic=None, options=options)
            self._compiled_finish = torch.compile(self._finish_block, dynamic=None, options=options)
        return self._compiled_projection, self._compiled_finish

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

    def _window_layout(self, values: torch.Tensor, shift: int):
        batch, frames, height, width = values.shape[:4]
        padded_height = ((height + shift + self.window - 1) // self.window) * self.window
        padded_width = ((width + shift + self.window - 1) // self.window) * self.window
        windows_h, windows_w = padded_height // self.window, padded_width // self.window
        layout_key = batch, frames, height, width, shift, values.device
        layout = self._window_layouts.get(layout_key)
        if layout is None:
            # Build the small row map once on CPU. Gather directly into the
            # final window layout; padded KV plus a permuted copy would double
            # the largest attention workspace at the decoder's memory peak.
            window = torch.arange(windows_h * windows_w)[:, None]
            token = torch.arange(frames * self.window**2)[None, :]
            yy = window.div(windows_w, rounding_mode="floor") * self.window + (token % self.window**2).div(self.window, rounding_mode="floor") - shift
            xx = (window % windows_w) * self.window + token % self.window - shift
            valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            temporal = token.div(self.window**2, rounding_mode="floor")
            local_index = (temporal * height + yy.clamp(0, height - 1)) * width + xx.clamp(0, width - 1)
            index = torch.arange(batch)[:, None, None] * frames * height * width + local_index
            layout = index.flatten().to(values.device), valid.to(values.device), local_index.to(values.device)
            self._window_layouts[layout_key] = layout
        return (batch, frames, height, width, padded_height, padded_width, windows_h, windows_w), layout

    def _partition(self, values: torch.Tensor, shift: int):
        shape, (index, valid, _) = self._window_layout(values, shift)
        batch, frames, _, _, _, _, windows_h, windows_w = shape
        tail = values.shape[4:]
        windows = values.flatten(0, 3).index_select(0, index).view(batch * windows_h * windows_w, frames * self.window**2, *tail)
        invalid = ~valid.repeat(batch, 1).view(batch * windows_h * windows_w, frames * self.window**2, *([1] * len(tail)))
        windows.masked_fill_(invalid, 0)
        return windows, valid, shape

    def _window_kv(self, values, suffix, shift):
        """Gather directly into KV storage with reserved register-token rows."""
        shape, (_, valid, index) = self._window_layout(values, shift)
        batch, frames, height, width, _, _, windows_h, windows_w = shape
        windows = windows_h * windows_w
        tokens, registers = frames * self.window**2, suffix.shape[1]
        tail = values.shape[4:]
        combined = values.new_empty((batch, windows, tokens + registers, *tail))
        destination = combined[:, :, :tokens]
        source = values.flatten(1, 3)[:, None].expand(batch, windows, frames * height * width, *tail)
        gather_index = index.view(1, windows, tokens, *([1] * len(tail))).expand(batch, windows, tokens, *tail)
        torch.gather(source, 2, gather_index, out=destination)
        destination.masked_fill_(~valid.view(1, windows, tokens, *([1] * len(tail))), 0)
        combined[:, :, tokens:].copy_(suffix[:, None])
        return combined.flatten(0, 1), valid

    def _unpartition(self, windows: torch.Tensor, shape: tuple[int, ...], shift: int) -> torch.Tensor:
        batch, frames, height, width, padded_height, padded_width, windows_h, windows_w = shape
        tail = windows.shape[2:]
        values = windows.view(batch, windows_h, windows_w, frames, self.window, self.window, *tail)
        values = values.permute(0, 3, 1, 4, 2, 5, *range(6, 6 + len(tail)))
        values = values.reshape(batch, frames, padded_height, padded_width, *tail)
        return values[:, :, shift : shift + height, shift : shift + width]

    def _window_attention(self, query, keys, values, suffix_key, suffix_value, shift):
        """Own padded/window KV only until the attention result is ready."""
        batch = query.shape[0]
        suffix_count = suffix_key.shape[1]
        window_queries, _, shape = self._partition(query, shift)
        all_keys, valid = self._window_kv(keys, suffix_key, shift)
        all_values, _ = self._window_kv(values, suffix_value, shift)
        windows_per_sample = window_queries.shape[0] // batch
        mask = torch.cat((valid, torch.ones(windows_per_sample, suffix_count, dtype=torch.bool, device=query.device)), dim=1)
        mask = mask.repeat(batch, 1)[:, None, None, :]
        output = F.scaled_dot_product_attention(window_queries.transpose(1, 2), all_keys.transpose(1, 2), all_values.transpose(1, 2), attn_mask=mask).transpose(1, 2)
        return output, shape

    def _register_attention(self, query, keys, values, suffix_key, suffix_value):
        """Own global-register KV independently of the window workspace."""
        batch, frames, height, width = keys.shape[:4]
        all_keys = torch.cat((keys.reshape(batch, frames * height * width, self.heads, self.head_dim), suffix_key), dim=1).transpose(1, 2)
        all_values = torch.cat((values.reshape(batch, frames * height * width, self.heads, self.head_dim), suffix_value), dim=1).transpose(1, 2)
        return F.scaled_dot_product_attention(query.transpose(1, 2), all_keys, all_values).transpose(1, 2).reshape(batch, query.shape[1], 2048)

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
        project_block, finish_block = self._block_functions()
        for layer_id, block in enumerate(self.blocks):
            shift = self.window // 2 if layer_id % 2 else 0
            query, key, value, suffix_query, suffix_key, suffix_value = project_block(block, hidden, suffix)
            cos, sin = self._ids(start, count, height, width, shift, device)
            query = _apply_rotary(query.reshape(batch, count * height * width, self.heads, self.head_dim), cos, sin).view(batch, count, height, width, self.heads, self.head_dim)
            key = _apply_rotary(key.reshape(batch, count * height * width, self.heads, self.head_dim), cos, sin).view(batch, count, height, width, self.heads, self.head_dim)
            suffix_cos, suffix_sin = self._suffix_ids(start, suffix_count, device)
            suffix_query = _apply_rotary(suffix_query, suffix_cos, suffix_sin)
            suffix_key = _apply_rotary(suffix_key, suffix_cos, suffix_sin)
            keys = key if state["keys"][layer_id] is None else torch.cat((state["keys"][layer_id], key), dim=1)
            values = value if state["values"][layer_id] is None else torch.cat((state["values"][layer_id], value), dim=1)
            del key, value
            window_output, shape = self._window_attention(query, keys, values, suffix_key, suffix_value, shift)
            del query
            output = self._unpartition(window_output, shape, shift).reshape(batch, count, height, width, 2048)
            del window_output
            suffix_output = self._register_attention(suffix_query, keys, values, suffix_key, suffix_value)
            del suffix_query, suffix_key, suffix_value
            # A tail view retains the full six-latent concatenation (and a
            # first value view can retain fused QKV). Own exactly four rows.
            state["keys"][layer_id] = keys[:, -self.cache_latents :].clone(memory_format=torch.contiguous_format)
            state["values"][layer_id] = values[:, -self.cache_latents :].clone(memory_format=torch.contiguous_format)
            del keys, values
            hidden, suffix = finish_block(block, hidden, suffix, output, suffix_output)
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
