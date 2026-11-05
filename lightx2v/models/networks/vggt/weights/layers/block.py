# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from torch import nn

from lightx2v.models.networks.vggt.weights.layers.attention import AttentionWeights
from lightx2v.models.networks.vggt.weights.layers.layer_scale import LayerScaleWeights
from lightx2v.models.networks.vggt.weights.layers.mlp import MlpWeights


class BlockWeights(nn.Module):
    _inference_type = ("lightx2v.models.networks.vggt.infer.layers.block", "BlockInfer")

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=AttentionWeights,
        ffn_layer=MlpWeights,
        qk_norm=False,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, qk_norm=qk_norm, rope=rope)
        self.ls1 = LayerScaleWeights(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, bias=ffn_bias)
        self.ls2 = LayerScaleWeights(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.vggt.infer.layers.block import BlockInfer

        return BlockInfer(self).infer(*args, **kwargs)
