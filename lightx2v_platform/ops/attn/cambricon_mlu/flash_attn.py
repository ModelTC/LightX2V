import math

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

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, model_cls=None, **kwds):
        if len(q.shape) == 3:
            bs = 1
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif len(q.shape) == 4:
            bs = q.shape[0]
        softmax_scale = 1 / math.sqrt(q.shape[-1])
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
        )
        x = x.reshape(bs * max_seqlen_q, -1)
        return x
