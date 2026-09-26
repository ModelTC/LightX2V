import torch
import torch_br
from torch_br.utils.tensor_methods import Sbp

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

_BR166_DIE_NUM = 2


@PLATFORM_ATTN_WEIGHT_REGISTER("supa_sdpa")
class SupaSDPAWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        drop_rate=0,
        attn_mask=None,
        causal=False,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        pre_transposed=False,
        **kwargs,
    ):
        if pre_transposed:
            h_q, s_q, head_size = q.shape
        else:
            s_q, h_q, head_size = q.shape
            q = q.transpose(0, 1).contiguous()
            k = k.transpose(0, 1).contiguous()
            v = v.transpose(0, 1).contiguous()

        output = torch_br.sueager_scaled_dot_product_attention_fwd(
            q, k, v,
            mask=None,
            dropout_prob=0.0,
            is_causal=causal,
            scale=None,
            algorithm="FMHA",
        )

        if isinstance(output, (tuple, list)):
            output = output[0]

        if _BR166_DIE_NUM > 1:
            output = torch_br.supa_shape_transform_qkv_fwd(output, 1, s_q, h_q, head_size, Sbp.sb(2))
        else:
            output = output.view(h_q, s_q, head_size).transpose(0, 1).contiguous()
            output = output.view(s_q, h_q * head_size)
            return output

        return output.view(s_q, h_q * head_size)
