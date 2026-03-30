import torch
from loguru import logger

from .utils.sla_util import get_block_map
from .utils.sparge_util import block_map_ordinal_lut_triton, get_block_map_meansim

try:
    import flash_attn  # noqa: F401
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    logger.info("flash_attn_varlen_func not found, please install flash_attn2 first")
    flash_attn_varlen_func = None

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    logger.info("flash_attn_varlen_func_v3 not found, please install flash_attn3 first")
    flash_attn_varlen_func_v3 = None

try:
    from flash_attn.cute import flash_attn_func as flash_attn_func_v4
except ImportError:
    logger.info("flash_attn.cute not found, please install flashattention4 first")
    flash_attn_func_v4 = None


from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


@ATTN_WEIGHT_REGISTER("flash_attn2")
class FlashAttn2Weight(AttnWeightTemplate):
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
        if len(q.shape) == 3:
            bs = 1
        elif len(q.shape) == 4:
            bs = q.shape[0]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ).reshape(bs * max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("flash_attn3")
class FlashAttn3Weight(AttnWeightTemplate):
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
        if len(q.shape) == 3:
            bs = 1
        elif len(q.shape) == 4:
            bs = q.shape[0]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
        x = flash_attn_varlen_func_v3(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ).reshape(bs * max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("flash_attn4")
class FlashAttn4Weight(AttnWeightTemplate):
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
        if len(q.shape) == 3:
            bs = 1
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif len(q.shape) == 4:
            bs = q.shape[0]
        assert bs == 1, "flash_attn4 doesn't support flash_attn_varlen_func now. Just use it for batchsize = 1 for sure."
        x, _ = flash_attn_func_v4(
            q,
            k,
            v,
        )
        x = x.reshape(bs * max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("spas_flash_attn4")
class SparseFlashAttn4Weight(AttnWeightTemplate):
    sparsity_ratio = 0.8
    sparse_mode = "sla_mode"

    def __init__(self):
        self.config = {}
        self.topk = 1 - self.sparsity_ratio
        self.BLKQ, self.BLKK = 128, 128

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
        if len(q.shape) == 3:
            bs = 1
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif len(q.shape) == 4:
            bs = q.shape[0]
        assert bs == 1, "flash_attn4 doesn't support flash_attn_varlen_func now. Just use it for batchsize = 1 for sure."

        # (L, H, D) -> (B, L, H, D)
        qt = q.transpose(1, 2).contiguous()
        kt = k.transpose(1, 2).contiguous()
        if self.sparse_mode == "sla_mode":
            sparse_map, lut, real_topk = get_block_map(qt, kt, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)
        elif self.sparse_mode == "sparge_mode":
            smooth_k = kt - kt.mean(dim=-2, keepdim=True)
            sparse_map = get_block_map_meansim(qt, smooth_k, cdfthreshd=None, topk=self.topk, return_lut=False, BLKQ=self.BLKQ, BLKK=self.BLKK)
        else:
            logger.info(f"spas_flash_attn4 sparse_mode only support sla_mode and sparge_mode now.")

        # (B, H, Q_block_num, K_block_num)
        full_block_idx, full_block_cnt = block_map_ordinal_lut_triton(sparse_map)
        mask_block_cnt = torch.zeros_like(full_block_cnt)
        mask_block_idx = torch.zeros_like(full_block_idx)

        x, _ = flash_attn_func_v4(
            q=q,
            k=k,
            v=v,
            mask_block_cnt=mask_block_cnt,
            mask_block_idx=mask_block_idx,
            full_block_cnt=full_block_cnt,
            full_block_idx=full_block_idx,
            block_size=(self.BLKQ, self.BLKK),
        )

        x = x.reshape(bs * max_seqlen_q, -1)
        return x
