# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn

from ..infer.attention import attention
from .bricks import get_norm_layer
from .encoders import MLP, MLPEncoder, TimestepEmbeddingEncoder
from .modulate_layers import ModulateDiT, apply_gate


class IndividualTokenRefinerBlock(nn.Module):
    def __init__(self, feat_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, mlp_act_type: str = "silu", qk_norm_type: str = "layer", qkv_bias: bool = True) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        assert self.feat_dim % num_heads == 0, f"feat_dim {self.feat_dim} must be divisible by num_heads {num_heads}"
        self.head_dim = feat_dim // num_heads
        self.mlp_hidden_dim = int(feat_dim * mlp_ratio)
        self.norm1 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=True, eps=1e-06)
        self.self_attn_qkv = nn.Linear(feat_dim, feat_dim * 3, bias=qkv_bias)
        self.self_attn_q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-06)
        self.self_attn_k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-06)
        self.self_attn_proj = nn.Linear(feat_dim, feat_dim, bias=qkv_bias)
        self.norm2 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=True, eps=1e-06)
        self.mlp = MLP(in_dim=feat_dim, feat_dim=self.mlp_hidden_dim, act_type=mlp_act_type, drop=dropout)
        self.adaLN_modulation = ModulateDiT(feat_dim=feat_dim, factor=2, act_type="silu")

    def forward(self, x: Tensor, c: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.token_refiner import IndividualTokenRefinerBlock_forward

        return IndividualTokenRefinerBlock_forward(self, x, c, attn_mask)


class IndividualTokenRefiner(nn.Module):
    def __init__(
        self, feat_dim: int, num_heads: int, num_layers: int, mlp_ratio: float = 4.0, dropout: float = 0.0, mlp_act_type: str = "silu", qk_norm_type: str = "layer", qkv_bias: bool = True
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(feat_dim=feat_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, mlp_act_type=mlp_act_type, qk_norm_type=qk_norm_type, qkv_bias=qkv_bias)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, c: Tensor, mask: Tensor | None = None) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.token_refiner import IndividualTokenRefiner_forward

        return IndividualTokenRefiner_forward(self, x, c, mask)


class SingleTokenRefiner(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feat_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        mlp_act_type: str = "silu",
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        attn_mode: str = "torch",
        **kwargs,
    ) -> None:
        super().__init__()
        self.attn_mode = attn_mode
        assert self.attn_mode == "torch", "Only support 'torch' mode for token refiner."
        self.input_embedder = nn.Linear(input_dim, feat_dim, bias=True)
        self.context_encoder = MLPEncoder(in_dim=feat_dim, feat_dim=feat_dim, num_layers=2, act_type=mlp_act_type)
        self.timestep_encoder = TimestepEmbeddingEncoder(embedding_dim=feat_dim, feat_dim=feat_dim, act_type=mlp_act_type)
        self.individual_token_refiner = IndividualTokenRefiner(
            feat_dim=feat_dim, num_heads=num_heads, num_layers=num_layers, mlp_ratio=mlp_ratio, dropout=dropout, mlp_act_type=mlp_act_type, qk_norm_type=qk_norm_type, qkv_bias=qkv_bias
        )

    def forward(self, x: Tensor, t: Tensor, mask: Tensor | None = None) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.token_refiner import SingleTokenRefiner_forward

        return SingleTokenRefiner_forward(self, x, t, mask)


__all__ = ["Optional", "Tensor", "apply_gate", "attention", "rearrange", "torch"]
