import math

import torch

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

try:
    import torch_mlu_ops as tmo
except ImportError:
    tmo = None


@torch.library.custom_op(
    "lightx2v::mlu_sage_attention",
    mutates_args=(),
    device_types="mlu",
)
def _mlu_sage_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    return tmo.sage_attn(
        q=q,
        k=k,
        v=v,
        cu_seq_lens_q=None,
        cu_seq_lens_kv=None,
        max_seq_len_q=max_seqlen_q,
        max_seq_len_kv=max_seqlen_kv,
        softmax_scale=softmax_scale,
        is_causal=causal,
        compute_dtype=torch.bfloat16,
    )


@_mlu_sage_attention.register_fake
def _mlu_sage_attention_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    return v.new_empty((*q.shape[:-1], v.shape[-1]))


@PLATFORM_ATTN_WEIGHT_REGISTER("mlu_sage_attn")
class MluSageAttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}
        assert tmo is not None, "torch_mlu_ops is not installed."

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, **kwds):
        if len(q.shape) == 3:
            bs = 1
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif len(q.shape) == 4:
            bs = q.shape[0]
        softmax_scale = kwds.get("softmax_scale", None)
        if softmax_scale is None:
            softmax_scale = 1 / math.sqrt(q.shape[-1])
        causal = kwds.get("causal", False)
        x = _mlu_sage_attention(q, k, v, max_seqlen_q, max_seqlen_kv, softmax_scale, causal)
        x = x.reshape(bs * max_seqlen_q, -1)
        return x
