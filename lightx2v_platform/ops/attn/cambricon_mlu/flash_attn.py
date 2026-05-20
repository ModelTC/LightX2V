import math

import torch

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

try:
    import torch_mlu_ops as tmo
except ImportError:
    tmo = None


@PLATFORM_ATTN_WEIGHT_REGISTER("mlu_flash_attn")
class MluFlashAttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}
        assert tmo is not None, "torch_mlu_ops is not installed."

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, **kwds):
        if q.dim() == 3:
            bs = 1 if cu_seqlens_q is None else cu_seqlens_q.shape[0] - 1
        elif q.dim() == 4:
            bs = q.shape[0]
        else:
            raise ValueError(f"mlu_flash_attn: unsupported q shape {tuple(q.shape)}, expected 3D or 4D")

        if max_seqlen_q is None:
            raise ValueError("mlu_flash_attn: max_seqlen_q is required")

        total_seqlen = bs * max_seqlen_q

        if bs == 1:
            if q.dim() == 3:
                q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
            cu_seqlens_q = None
            cu_seqlens_kv = None
        else:
            if cu_seqlens_q is None:
                cu_seqlens_kv = None
            else:
                if cu_seqlens_q.is_cpu:
                    cu_seqlens_q = cu_seqlens_q.to(q.device, non_blocking=True)
                if cu_seqlens_kv is not None and cu_seqlens_kv.is_cpu:
                    cu_seqlens_kv = cu_seqlens_kv.to(k.device, non_blocking=True)
                if q.dim() == 4:
                    q = q.reshape(-1, q.shape[-2], q.shape[-1])
                    k = k.reshape(-1, k.shape[-2], k.shape[-1])
                    v = v.reshape(-1, v.shape[-2], v.shape[-1])

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        softmax_scale = 1 / math.sqrt(q.shape[-1])
        compute_dtype = torch.bfloat16 if q.dtype == torch.bfloat16 else torch.half
        x = tmo.flash_attention(
            q=q,
            k=k,
            v=v,
            cu_seq_lens_q=cu_seqlens_q,
            cu_seq_lens_kv=cu_seqlens_kv,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_kv=max_seqlen_kv,
            softmax_scale=softmax_scale,
            return_lse=False,
            out_dtype=q.dtype,
            is_causal=False,
            out=None,
            alibi_slope=None,
            attn_bias=None,
            compute_dtype=compute_dtype,
        )
        return x.reshape(total_seqlen, -1)
