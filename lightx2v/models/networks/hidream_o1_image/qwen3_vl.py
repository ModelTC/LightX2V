import os

import torch
import torch.nn as nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

USE_BF16_ROPE = os.environ.get("USE_BF16_ROPE", "0")


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen3VLTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3VLTextConfig, device=None):
        super().__init__()
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "linear"))
        if rope_type == "default":
            rope_type = "linear"
        if rope_type == "linear" and "factor" not in rope_scaling:
            rope_scaling = {**rope_scaling, "factor": 1.0}

        self.rope_type = rope_type
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.config.rope_scaling = rope_scaling
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self.mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        if getattr(self.inv_freq, "is_meta", False) or getattr(self.original_inv_freq, "is_meta", False):
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device=x.device)
            self.inv_freq = inv_freq
            self.original_inv_freq = inv_freq

        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if USE_BF16_ROPE == "1":
            inv_freq_expanded = self.inv_freq[None, None, :, None].float().to(device=x.device).expand(3, position_ids.shape[1], -1, 1)
        else:
            inv_freq_expanded = self.original_inv_freq[None, None, :, None].float().to(device=x.device).expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        # with torch.autocast(device_type=device_type, cache_enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
