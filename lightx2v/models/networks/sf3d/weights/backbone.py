# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from lightx2v.models.networks.sf3d.utils.base import BaseModule
from lightx2v.models.networks.three_models.ops import Attention

_registered_attention = Attention()


class GEGLU(nn.Module):
    """
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        from lightx2v.models.networks.sf3d.infer.backbone import GEGLU_gelu

        return GEGLU_gelu(self, gate)

    def forward(self, hidden_states, scale: float = 1.0):
        from lightx2v.models.networks.sf3d.infer.backbone import GEGLU_forward

        return GEGLU_forward(self, hidden_states, scale)


class CrossAttention(nn.Module):
    def __init__(self, dim, kv_dim=None, num_heads=16, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        kv_dim = dim if not kv_dim else kv_dim
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        from lightx2v.models.networks.sf3d.infer.backbone import CrossAttention_forward

        return CrossAttention_forward(self, x_q, x_kv)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        act_fn = GEGLU(dim, inner_dim)
        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from lightx2v.models.networks.sf3d.infer.backbone import FeedForward_forward

        return FeedForward_forward(self, x)


class BasicBlock(nn.Module):
    def __init__(self, dim: int, kv_dim: int | None = None, num_heads: int = 16, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0, ff_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, kv_dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, kv_dim=kv_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=ff_drop)

    def forward(self, z, x):
        from lightx2v.models.networks.sf3d.infer.backbone import BasicBlock_forward

        return BasicBlock_forward(self, z, x)


class SingleStreamTransformer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        num_attention_heads: int = 16
        attention_head_dim: int = 88
        in_channels: int | None = None
        out_channels: int | None = None
        num_layers: int = 16
        dropout: float = 0.0
        norm_num_groups: int = 32
        cross_attention_dim: int | None = None
        attention_bias: bool = False

    cfg: Config

    def configure(self) -> None:
        self.num_attention_heads = self.cfg.num_attention_heads
        self.attention_head_dim = self.cfg.attention_head_dim
        inner_dim = self.num_attention_heads * self.attention_head_dim
        self.norm = torch.nn.GroupNorm(num_groups=self.cfg.norm_num_groups, num_channels=self.cfg.in_channels, eps=1e-06, affine=True)
        self.proj_in = nn.Linear(self.cfg.in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicBlock(inner_dim, kv_dim=self.cfg.cross_attention_dim, num_heads=self.num_attention_heads, qkv_bias=self.cfg.attention_bias, proj_drop=self.cfg.dropout, ff_drop=self.cfg.dropout)
                for d in range(self.cfg.num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, self.cfg.in_channels)

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        from lightx2v.models.networks.sf3d.infer.backbone import SingleStreamTransformer_forward

        return SingleStreamTransformer_forward(self, hidden_states, encoder_hidden_states, **kwargs)


class FuseBlock(nn.Module):
    """
    Fuse X in to Z with cross attention
    """

    def __init__(self, dim_z: int, dim_x: int, num_heads: int = 16, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0, ff_drop: float = 0.0, norm_x_input: bool = True):
        super().__init__()
        self.norm_x_input = norm_x_input
        if self.norm_x_input:
            self.norm_x = nn.LayerNorm(dim_x)
        self.attn = CrossAttention(dim_z, kv_dim=dim_x, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm_z1 = nn.LayerNorm(dim_z)
        self.norm_z2 = nn.LayerNorm(dim_z)
        self.ff = FeedForward(dim_z, dropout=ff_drop)

    def forward(self, z, x):
        from lightx2v.models.networks.sf3d.infer.backbone import FuseBlock_forward

        return FuseBlock_forward(self, z, x)


@torch.no_grad()
def get_triplane_attention_mask(res):
    N = 3 * res * res
    attn_mask = torch.zeros(3, res, res, 3, res, res)
    i, j = torch.meshgrid(torch.arange(res), torch.arange(res))
    attn_mask[0, i, j, 1, i, :] = 1.0
    attn_mask[0, i, j, 2, j, :] = 1.0
    attn_mask[1, i, j, 0, i, :] = 1.0
    attn_mask[1, i, j, 2, :, j] = 1.0
    attn_mask[2, i, j, 0, :, i] = 1.0
    attn_mask[2, i, j, 1, :, j] = 1.0
    attn_mask = attn_mask.bool()
    attn_bias = torch.empty_like(attn_mask, dtype=torch.float)
    attn_bias.masked_fill_(attn_mask, 0.0)
    attn_bias.masked_fill_(~attn_mask, float("-inf"))
    return attn_bias.reshape(N, N)


class TriplaneAttention(nn.Module):
    def __init__(self, dim: int, resolution: int, num_heads: int = 16, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0, full_attention: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.resolution = resolution
        self.full_attention = full_attention
        self.attn_mask = get_triplane_attention_mask(resolution) if not full_attention else None

    def forward(self, x):
        from lightx2v.models.networks.sf3d.infer.backbone import TriplaneAttention_forward

        return TriplaneAttention_forward(self, x)


class TwoStreamBlock(nn.Module):
    def __init__(
        self,
        dim_latent: int,
        dim_input: int,
        num_basic_blocks: int = 4,
        num_heads: int = 16,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ff_drop: float = 0.0,
        norm_x_input: bool = True,
        dim_cross: int | None = None,
    ):
        super().__init__()
        self.fuse_block_in = FuseBlock(dim_latent, dim_input, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, ff_drop=ff_drop, norm_x_input=norm_x_input)
        self.transformer_block = nn.ModuleList(
            [BasicBlock(dim_latent, kv_dim=dim_cross, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop, ff_drop=ff_drop) for _ in range(num_basic_blocks)]
        )
        self.fuse_block_out = FuseBlock(dim_input, dim_latent, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, ff_drop=ff_drop, norm_x_input=norm_x_input)

    def forward(self, latent, input, cross_input):
        from lightx2v.models.networks.sf3d.infer.backbone import TwoStreamBlock_forward

        return TwoStreamBlock_forward(self, latent, input, cross_input)


class TwoStreamInterleaveTransformer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        num_attention_heads: int = 16
        attention_head_dim: int = 64
        raw_triplane_channels: int = 1024
        triplane_channels: int = 1024
        raw_image_channels: int = 1024
        num_latents: int = 1792
        num_blocks: int = 4
        num_basic_blocks: int = 3
        dropout: float = 0.0
        latent_init_std: float = 0.02
        norm_num_groups: int = 32
        attention_bias: bool = False
        norm_x_input: bool = False
        cross_attention_dim: int = 1024
        mix_latent: bool = True

    cfg: Config

    def configure(self) -> None:
        self.mix_latent = self.cfg.mix_latent
        self.num_attention_heads = self.cfg.num_attention_heads
        self.attention_head_dim = self.cfg.attention_head_dim
        self.num_latents = self.cfg.num_latents
        self.latent_dim = self.num_attention_heads * self.attention_head_dim
        if self.cfg.norm_num_groups > 0:
            self.norm_triplane = torch.nn.GroupNorm(num_groups=self.cfg.norm_num_groups, num_channels=self.cfg.raw_triplane_channels, eps=1e-06, affine=True)
        else:
            self.norm_triplane = nn.LayerNorm(self.cfg.raw_triplane_channels)
        self.proj_triplane = nn.Linear(self.cfg.raw_triplane_channels, self.cfg.triplane_channels)
        if self.mix_latent:
            self.norm_image = nn.LayerNorm(self.cfg.raw_image_channels)
            self.proj_image = nn.Linear(self.cfg.raw_image_channels, self.latent_dim)
        self.norm_latent = nn.LayerNorm(self.latent_dim)
        self.proj_latent = nn.Linear(self.latent_dim, self.latent_dim)
        self.latent_init = nn.Parameter(torch.zeros(1, self.num_latents, self.latent_dim))
        nn.init.normal_(self.latent_init, std=self.cfg.latent_init_std)
        self.main_blocks = nn.ModuleList(
            [
                TwoStreamBlock(
                    self.latent_dim,
                    self.cfg.triplane_channels,
                    num_basic_blocks=self.cfg.num_basic_blocks,
                    num_heads=self.num_attention_heads,
                    qkv_bias=self.cfg.attention_bias,
                    proj_drop=self.cfg.dropout,
                    ff_drop=self.cfg.dropout,
                    norm_x_input=self.cfg.norm_x_input,
                    dim_cross=self.cfg.cross_attention_dim,
                )
                for _ in range(self.cfg.num_blocks)
            ]
        )
        self.proj_out = nn.Linear(self.cfg.triplane_channels, self.cfg.raw_triplane_channels)

    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        from lightx2v.models.networks.sf3d.infer.backbone import TwoStreamInterleaveTransformer_forward

        return TwoStreamInterleaveTransformer_forward(self, hidden_states, encoder_hidden_states, **kwargs)


__all__ = ["F", "nn", "torch"]


def _registered_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, scale=None):
    if dropout_p:
        raise ValueError("SF3D is inference-only")
    return _registered_attention(q, k, v, mask=attn_mask, scale=scale)
