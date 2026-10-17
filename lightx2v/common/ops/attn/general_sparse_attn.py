from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, SPARSE_MASK_GENERATOR_REGISTER, SPARSE_OPERATOR_REGISTER

from .sparse_mask_generator import SlaMaskGenerator
from .template import AttnWeightTemplate


@ATTN_WEIGHT_REGISTER("general_sparse_attn")
class GeneralSparseAttnWeight(AttnWeightTemplate):
    sparse_mask_generator = None
    sparse_operator = None
    sparse_setting = {}
    operator_setting = {}
    attnmap_frame_num = None

    def __init__(self):
        self.config = {}

        self._setup_operator()
        self._setup_mask_generator()

        logger.info(
            f"GeneralSparseAttnWeight: sparse_setting={self.sparse_setting}, operator_setting={self.operator_setting}, operator={self.sparse_operator}, mask_generator={self.sparse_mask_generator}, attnmap_frame_num={self.attnmap_frame_num}"
        )

    def _setup_operator(self):
        self.operator = SPARSE_OPERATOR_REGISTER[self.sparse_operator](operator_setting=self.operator_setting)

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
        if isinstance(self.mask_generator, SlaMaskGenerator) and hasattr(self.operator, "apply_sla") and q.is_cuda:
            # Keep the shape-dependent router within the CUDA custom op. Python
            # top-k sizing otherwise specializes each growing AR history length,
            # and compiler fusion can alter the router's BF16 rounding.
            return self.operator.apply_sla(q, k, v, self.mask_generator.topk_ratio)
        if isinstance(self.mask_generator, SlaMaskGenerator) and hasattr(self.operator, "apply_lut"):
            if getattr(self.operator, "reuse_prepared_qk", False):
                mask, lut, topk, prepared_qk = self.mask_generator.prepare(q, k, return_qk=True)
                kwargs["prepared_qk"] = prepared_qk
            else:
                mask, lut, topk = self.mask_generator.prepare(q, k)
            return self.operator.apply_lut(
                q,
                k,
                v,
                mask,
                lut,
                topk,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                **kwargs,
            )

        # Generate sparse mask
        mask = self.mask_generator(q, k)

        # reorg
        q, k, v = self.mask_generator.reorg(q, k, v)

        # Apply sparse operator
        out = self.operator(q, k, v, mask, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv, **kwargs)

        # restore
        out = self.mask_generator.restore(out)

        return out
