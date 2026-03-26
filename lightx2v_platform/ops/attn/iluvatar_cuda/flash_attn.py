import math

import torch

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

try:
    from ixformer.contrib.vllm_flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


@PLATFORM_ATTN_WEIGHT_REGISTER("iluvatar_flash_attn")
class IluvatarFlashAttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}
        assert flash_attn_varlen_func is not None, "iluvatar ixformer is not installed."

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, **kwds):
        half_dtypes = (torch.float16, torch.bfloat16)
        device = q.device
        dtype =  q.dtype

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        if len(q.shape) == 3:
            bs = 1
        elif len(q.shape) == 4:
            bs, lq, lk = q.size(0), q.size(1), k.size(1)
            # preprocess query
            if cu_seqlens_q is None:
                q = half(q.flatten(0, 1))
                cu_seqlens_q = torch.tensor([lq] * bs, dtype=torch.int32).to(device=q.device, non_blocking=True)
                cu_seqlens_q = torch.cat([cu_seqlens_q.new_zeros([1]), cu_seqlens_q]).cumsum(0, dtype=torch.int32)
            else:
                q = half(torch.cat([u[:v] for u, v in zip(q, cu_seqlens_q)]))
            # preprocess key, value
            if cu_seqlens_kv is None:
                k = half(k.flatten(0, 1))
                v = half(v.flatten(0, 1))
                cu_seqlens_kv = torch.tensor([lk] * bs, dtype=torch.int32).to(device=k.device, non_blocking=True)
                cu_seqlens_kv = torch.cat([cu_seqlens_kv.new_zeros([1]), cu_seqlens_kv]).cumsum(0, dtype=torch.int32)
            else:
                k = half(torch.cat([u[:v] for u, v in zip(k, cu_seqlens_kv)]))
                v = half(torch.cat([u[:v] for u, v in zip(v, cu_seqlens_kv)]))

        q = q.to(v.dtype)
        k = k.to(v.dtype)
        softmax_scale = 1 / math.sqrt(q.shape[-1])
        x = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q.to(device),  # cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv.to(device),  # cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_kv,
            softmax_scale=softmax_scale,
            return_softmax_lse=False,
            causal=False,
        )
        return x.reshape(bs * max_seqlen_q, -1)
