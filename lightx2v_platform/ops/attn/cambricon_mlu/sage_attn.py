import math

import torch

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

try:
    import torch_mlu_ops as tmo
except ImportError:
    tmo = None


@PLATFORM_ATTN_WEIGHT_REGISTER("mlu_sage_attn")
class MluSageAttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}
        assert tmo is not None, "torch_mlu_ops is not installed."

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, model_cls=None):
        if len(q.shape) == 3:
            bs = 1
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif len(q.shape) == 4:
            bs = q.shape[0]
        softmax_scale = 1 / math.sqrt(q.shape[-1])
        x = tmo.sage_attn(
            q=q, k=k, v=v, cu_seq_lens_q=None, cu_seq_lens_kv=None, max_seq_len_kv=max_seqlen_kv, max_seq_len_q=max_seqlen_q, is_causal=False, compute_dtype=torch.bfloat16, softmax_scale=softmax_scale
        )
        x = x.reshape(bs * max_seqlen_q, -1)
        return x
