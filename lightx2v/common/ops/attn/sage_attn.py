import torch
from loguru import logger
import math

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate

if torch.cuda.is_available() and torch.cuda.get_device_capability(0) in [(8, 9), (12, 0)]:
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_triton as sageattn
    except ImportError:
        logger.info("sageattn not found, please install sageattention first")
        sageattn = None
else:
    try:
        from sageattention import sageattn
    except ImportError:
        logger.info("sageattn not found, please install sageattention first")
        sageattn = None

try:
    from sageattn3 import sageattn3_blackwell
except ImportError:
    logger.info("sageattn3 not found, please install sageattention first")
    sageattn3_blackwell = None

try:
    import torch_mlu_ops as tmo
except ImportError:
    tmo = None
    logger.info("torch_mlu_ops not found.")


@ATTN_WEIGHT_REGISTER("sage_attn2")
class SageAttn2Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        model_cls=None,
    ):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if len(q.shape) == 3:
            bs = 1
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif len(q.shape) == 4:
            bs = q.shape[0]
        x = sageattn(
            q,
            k,
            v,
            tensor_layout="NHD",
        ).view(bs * max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("sage_attn3")
class SageAttn3Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        model_cls=None,
    ):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if len(q.shape) == 3:
            bs = 1
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif len(q.shape) == 4:
            bs = q.shape[0]

        x = sageattn3_blackwell(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).reshape(bs * max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("mlu_sage_attn")
class MluSageAttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        model_cls=None,
        **kws
    ):
        if len(q.shape) == 3:
            bs = 1
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif len(q.shape) == 4:
            bs = q.shape[0]
        softmax_scale = 1 / math.sqrt(q.shape[-1])
        x = tmo.sage_attn(q=q, k=k, v=v, cu_seq_lens_q=None, cu_seq_lens_kv=None, max_seq_len_kv=max_seqlen_kv, max_seq_len_q=max_seqlen_q, is_causal=False, compute_dtype=torch.bfloat16,softmax_scale=softmax_scale)
        x = x.reshape(bs * max_seqlen_q, -1)
        return x
