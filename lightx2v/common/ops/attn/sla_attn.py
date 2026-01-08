from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate

from .utils.sla_util import get_block_map
from .kernels.sla_kernel import _attention


@ATTN_WEIGHT_REGISTER("sla_attn")
class SlaAttnWeight(AttnWeightTemplate):

    sparsity_ratio = 0.8
    operator = 'triton'

    def __init__(self):
        self.config = {}

        self.topk = 1 - self.sparsity_ratio
        if self.operator == 'triton':
            self.BLKQ, self.BLKK = 64, 64

        logger.info(f"SlaAttnWeight: sparsity_ratio={self.sparsity_ratio}, operator={self.operator}, topk={self.topk}, BLKQ={self.BLKQ}, BLKK={self.BLKK}")


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
        # (L, H, D) -> (B, H, L, D)
        q = q.unsqueeze(0).transpose(1, 2).contiguous()
        k = k.unsqueeze(0).transpose(1, 2).contiguous()
        v = v.unsqueeze(0).transpose(1, 2).contiguous()
        
        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)

        out = _attention.apply(q, k, v, sparse_map, lut, real_topk, self.BLKQ, self.BLKK)
        out = out.transpose(1, 2).reshape(max_seqlen_q, -1)

        return out
