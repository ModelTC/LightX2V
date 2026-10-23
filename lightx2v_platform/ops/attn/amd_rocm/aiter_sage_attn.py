"""SageAttention on AMD Radeon RDNA4 (gfx1201) through aiter's ``gfx1201_sage_attention``.

INT8 Q K^T, FP8 P V, FP32 accumulation, for BF16 Q/K/V with head dim 128, non-causal, without mask.
Select with ``attn_type=aiter_sage_attn`` under ``PLATFORM=amd_rocm``; on other GPUs the backend
refuses to construct instead of falling back silently.
"""

import torch

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

_SUPPORTED_ARCHS = ("gfx1201",)


def _device_arch():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]


@PLATFORM_ATTN_WEIGHT_REGISTER("aiter_sage_attn")
class AiterSageAttnWeight(AttnWeightTemplate):
    """Dense SageAttention backed by ``aiter.gfx1201_sage_attention``."""

    def __init__(self):
        self.config = {}
        if getattr(torch.version, "hip", None) is None:
            raise RuntimeError("aiter_sage_attn requires AMD ROCm (torch.version.hip is not set).")
        arch = _device_arch()
        if arch not in _SUPPORTED_ARCHS:
            raise RuntimeError(f"aiter_sage_attn supports {', '.join(_SUPPORTED_ARCHS)}, but this GPU is {arch}. Use aiter_attn or torch_sdpa instead.")
        import aiter

        if not hasattr(aiter, "gfx1201_sage_attention"):
            raise ImportError("aiter_sage_attn requires an aiter build that provides gfx1201_sage_attention (ROCm/aiter#6390).")
        self.attention = aiter.gfx1201_sage_attention

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        if kwargs.get("causal") or kwargs.get("attn_mask") is not None or kwargs.get("drop_rate"):
            raise ValueError("aiter_sage_attn supports non-causal attention without mask or dropout only.")
        for cu_seqlens in (cu_seqlens_q, cu_seqlens_kv):
            if cu_seqlens is not None and cu_seqlens.numel() > 2:
                raise ValueError("aiter_sage_attn does not support packed variable-length sequences.")
        if q.ndim == 3:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif q.ndim != 4:
            raise ValueError(f"aiter_sage_attn expects 3D or 4D Q/K/V, got q.ndim={q.ndim}.")
        out = self.attention(q.contiguous(), k.contiguous(), v.contiguous(), softmax_scale=kwargs.get("softmax_scale"))
        return out.reshape(out.shape[0] * out.shape[1], -1)
