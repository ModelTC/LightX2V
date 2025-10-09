import torch
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate

if torch.cuda.get_device_capability(0) == (8, 9):
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
