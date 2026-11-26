import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    flash_attn_varlen_func_v3 = None


class TorchAttention(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        assert len(args) == 0 or len(args) > 2, "query, key should both provided by args / kwargs"
        q = kwargs.get("query") or args[0]
        k = kwargs.get("key") or args[1]
        b, h, sq, d = q.shape
        b, h, sk, d = k.shape
        return b * h * (4 * d * (sq / 1e6) * (sk / 1e6))

    def forward(self, *args, **kwargs):
        return F.scaled_dot_product_attention(*args, **kwargs)


class FlashAttentionVarlen(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        _, h, d = output.shape
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
        return h * (4 * d * (seqlens_q * seqlens_k).sum())

    def forward(self, *args, **kwargs):
        kwargs["deterministic"] = torch.are_deterministic_algorithms_enabled()
        if flash_attn_varlen_func is not None:
            return flash_attn_varlen_func(*args, **kwargs)
        if flash_attn_varlen_func_v3 is not None:
            # FlashAttention-3 (Hopper) has no dropout and may also return the softmax LSE.
            kwargs.pop("dropout_p", None)
            out = flash_attn_varlen_func_v3(*args, **kwargs)
            return out[0] if isinstance(out, tuple) else out
        raise ImportError("SeedVR attention requires flash_attn (FlashAttention-2) or flash_attn_interface (FlashAttention-3).")
