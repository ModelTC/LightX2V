# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Vendored from Wan2.2/wan/modules/s2v for bit-aligned S2V audio encoding.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode="replicate", **kwargs):
        super().__init__()
        self.pad_mode = pad_mode
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class MotionEncoder_tc(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads=int, need_global=True, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.num_heads = num_heads
        self.need_global = need_global
        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1)
        if need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1)
        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)
        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        x = rearrange(x, "b t c -> b c t")
        x_ori = x.clone()
        b, _, _ = x.shape
        x = self.conv1_local(x)
        x = rearrange(x, "b (n c) t -> (b n) t c", n=self.num_heads)
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv2(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv3(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm3(x)
        x = self.act(x)
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        if not self.need_global:
            return x_local

        x = self.conv1_global(x_ori)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv2(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv3(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm3(x)
        x = self.act(x)
        x = self.final_linear(x)
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        return x, x_local


class CausalAudioEncoder(nn.Module):
    def __init__(self, dim=1024, num_layers=25, out_dim=5120, num_token=4, need_global=False):
        super().__init__()
        self.encoder = MotionEncoder_tc(in_dim=dim, hidden_dim=out_dim, num_heads=num_token, need_global=need_global)
        self.weights = nn.Parameter(torch.ones((1, num_layers, 1, 1)) * 0.01)
        self.act = nn.SiLU()

    def forward(self, features):
        with torch.amp.autocast("cuda", dtype=torch.float32):
            weights = self.act(self.weights)
            weights_sum = weights.sum(dim=1, keepdims=True)
            weighted_feat = ((features * weights) / weights_sum).sum(dim=1)
            weighted_feat = weighted_feat.permute(0, 2, 1)
            return self.encoder(weighted_feat)
