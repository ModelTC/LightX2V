from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, SPARSE_MASK_GENERATOR_REGISTER, SPARSE_OPERATOR_REGISTER

from .template import AttnWeightTemplate


@ATTN_WEIGHT_REGISTER("general_sparse_attn")
class GeneralSparseAttnWeight(AttnWeightTemplate):
    sparse_mask_generator = None
    sparse_operator = None
    sparse_setting = {}
    attnmap_frame_num = None

    def __init__(self):
        self.config = {}

        self._setup_operator()
        self._setup_mask_generator()

        logger.info(
            f"GeneralSparseAttnWeight: sparse_setting={self.sparse_setting}, operator={self.sparse_operator}, mask_generator={self.sparse_mask_generator}, attnmap_frame_num={self.attnmap_frame_num}"
        )

    def _setup_operator(self):
        self.operator = SPARSE_OPERATOR_REGISTER[self.sparse_operator]()

    def _setup_mask_generator(self):
        self.mask_generator = SPARSE_MASK_GENERATOR_REGISTER[self.sparse_mask_generator](self.operator.q_block_size, self.operator.k_block_size, self.sparse_setting, self.attnmap_frame_num)

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
        # Generate sparse mask
        mask = self.mask_generator(q, k)

        # reorg
        q, k, v = self.mask_generator.reorg(q, k, v)

        # Apply sparse operator
        out = self.operator(q, k, v, mask, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, **kwargs)

        # restore
        out = self.mask_generator.restore(out)

        return out
