from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v_platform.ops.attn.template import AttnWeightTemplate

try:
    from sageattention import sageattn
except ImportError:
    logger.info("sageattn not found, please install sageattention first")
    sageattn = None


@ATTN_WEIGHT_REGISTER("metax_sage_attn2")
class MetaxSageAttn2Weight(AttnWeightTemplate):
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
        **kwargs,
    ):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if len(q.shape) == 3:
            bs = 1
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif len(q.shape) == 4:
            bs = q.shape[0]
        x = (
            sageattn(
                q,
                k,
                v,
                tensor_layout="NHD",
            )[0]
            .view(bs * max_seqlen_q, -1)
            .type(q.dtype)
        )
        return x
