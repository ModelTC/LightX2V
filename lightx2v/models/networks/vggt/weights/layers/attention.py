# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from torch import nn

from lightx2v.common.ops.inference_module import InferenceAttention


class AttentionWeights(nn.Module):
    _inference_type = ("lightx2v.models.networks.vggt.infer.layers.attention", "AttentionInfer")

    def __init__(self, dim, num_heads=8, qkv_bias=True, proj_bias=True, norm_layer=nn.LayerNorm, qk_norm=False, rope=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = rope
        self.attention = InferenceAttention("torch_sdpa")

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.vggt.infer.layers.attention import AttentionInfer

        return AttentionInfer(self).infer(*args, **kwargs)
