# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3

    _USE_FLASH_ATTN_V3 = True
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2

    _USE_FLASH_ATTN_V3 = False
from ...comm.communication import _All2All
from ...comm.padding import depad_by_length, pad_by_length


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def _compute_qkv(self, x: Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        return q, k, v, B, N, C

    def _apply_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if q.dtype == torch.bfloat16 or q.dtype == torch.float16:
            if q.is_contiguous():
                q = q.transpose(1, 2)
            else:
                q = q.transpose(1, 2).contiguous()
            if k.is_contiguous():
                k = k.transpose(1, 2)
            else:
                k = k.transpose(1, 2).contiguous()
            if v.is_contiguous():
                v = v.transpose(1, 2)
            else:
                v = v.transpose(1, 2).contiguous()
            if _USE_FLASH_ATTN_V3:
                x = flash_attn_func_v3(q, k, v)
            else:
                x = flash_attn_func_v2(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            if x.is_contiguous():
                x = x.transpose(1, 2)
            else:
                x = x.transpose(1, 2).contiguous()
        else:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        return x

    def _project_output(self, x: Tensor, B: int, N: int, C: int) -> Tensor:
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: Tensor, pos=None) -> Tensor:
        q, k, v, B, N, C = self._compute_qkv(x)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        x = self._apply_attention(q, k, v)
        return self._project_output(x, B, N, C)


class DistAttention(Attention):
    def forward(self, x: Tensor, pos=None, sp_size=1, sp_group=None, padding_tokens=0) -> Tensor:
        q, k, v, B, N, C = self._compute_qkv(x)

        if sp_size > 1:
            q = _All2All.apply(q, 1, 2, sp_group, False)
            k = _All2All.apply(k, 1, 2, sp_group, False)
            v = _All2All.apply(v, 1, 2, sp_group, False)
            q = depad_by_length(q, padding_tokens, 2)
            k = depad_by_length(k, padding_tokens, 2)
            v = depad_by_length(v, padding_tokens, 2)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        x = self._apply_attention(q, k, v)

        if sp_size > 1:
            x = pad_by_length(x, padding_tokens, 2, 0)
            x = _All2All.apply(x, 2, 1, sp_group, False)

        return self._project_output(x, B, N, C)


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if attn_bias is not None:
            raise AssertionError("xFormers is required for using nested tensors")
        return super().forward(x)
