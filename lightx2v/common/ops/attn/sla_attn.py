import torch
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .kernels.sla_kernel import _attention
from .kernels.sla_kernel_ar import _attention_ar
from .template import AttnWeightTemplate
from .utils.sla_util import get_block_map, get_cuda_arch
from .utils.sla_util_blhd import get_block_map_blhd
from .utils.sparge_util import block_map_ordinal_lut_triton, block_map_incremental_lut_triton, sage2_block_sparse_attn
try:
    from flash_attn.cute import flash_attn_func as flash_attn_func_v4
except ImportError:
    logger.info("flash_attn.cute not found, please install flashattention4 first")
    flash_attn_func_v4 = None

try:
    from sageattn3_sparse import sage3_block_sparse_attn
except ImportError:
    logger.info("sageattn3_sparse not found, please install sageattn3_sparse first")
    sage3_block_sparse_attn = None

try:
    from magi_attention.functional import flex_flash_attn_func as magi_ffa_func
except ImportError:
    magi_ffa_func = None


@ATTN_WEIGHT_REGISTER("sla_attn")
class SlaAttnWeight(AttnWeightTemplate):
    sparsity_ratio = 0.8
    operator = "triton"
    per_block_mean = False

    def __init__(self):
        self.config = {}

        self.arch = get_cuda_arch(torch.cuda.current_device())
        self.topk = 1 - self.sparsity_ratio
        if self.operator == "triton":
            self.BLKQ, self.BLKK = 128, 128
            self.apply_func = self.apply_triton
        elif self.operator == "triton_ar":  # triton for AR models
            self.BLKQ, self.BLKK = 128, 128
            self.apply_func = self.apply_triton_ar
        elif self.operator == "sage2":
            if self.arch == "sm90":
                self.BLKQ, self.BLKK = 64, 128
            else:
                self.BLKQ, self.BLKK = 128, 64
            self.apply_func = self.apply_sage2
        elif self.operator == "sage3":
            self.BLKQ, self.BLKK = 128, 128
            self.apply_func = self.apply_sage3
        elif self.operator == "fa4":
            self.BLKQ, self.BLKK = 128, 128
            self.apply_func = self.apply_fa4
        elif self.operator == "magi":
            self.BLKQ, self.BLKK = 128, 128
            self.apply_func = self.apply_magi
        else:
            raise NotImplementedError(f"Not supported SLA operator: {self.operator}.")

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
        **kwargs,
    ):
        return self.apply_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, **kwargs)

    def apply_triton(
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
        # (L, H, D) -> (B, H, L, D)
        q = q.unsqueeze(0).transpose(1, 2).contiguous()
        k = k.unsqueeze(0).transpose(1, 2).contiguous()
        v = v.unsqueeze(0).transpose(1, 2).contiguous()

        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)

        out = _attention.apply(q, k, v, sparse_map, lut, real_topk, self.BLKQ, self.BLKK)
        out = out.transpose(1, 2).reshape(max_seqlen_q, -1)

        return out

    def apply_triton_ar(
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
        # (L, H, D) -> (B, H, L, D)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

        sparse_map, lut, real_topk = get_block_map_blhd(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)

        out = _attention_ar.apply(q, k, v, sparse_map, lut, real_topk, self.BLKQ, self.BLKK)
        out = out.reshape(max_seqlen_q, -1)

        return out

    def apply_sage2(
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
        # (L, H, D) -> (B, H, L, D)
        q = q.unsqueeze(0).transpose(1, 2).contiguous()
        k = k.unsqueeze(0).transpose(1, 2).contiguous()
        v = v.unsqueeze(0).transpose(1, 2).contiguous()

        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)
        lut, valid_block_num = block_map_incremental_lut_triton(sparse_map)

        out = sage2_block_sparse_attn(q, k, v, lut, valid_block_num, self.BLKQ, self.BLKK, self.arch)
        out = out.transpose(1, 2).reshape(max_seqlen_q, -1)
        return out

    def apply_sage3(
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
        # (L, H, D) -> (B, H, L, D)
        q = q.unsqueeze(0).transpose(1, 2).contiguous()
        k = k.unsqueeze(0).transpose(1, 2).contiguous()
        v = v.unsqueeze(0).transpose(1, 2).contiguous()

        sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)
        lut, valid_block_num = block_map_ordinal_lut_triton(sparse_map)
        out = sage3_block_sparse_attn(q, k, v, lut, valid_block_num, per_block_mean=self.per_block_mean)
        out = out.transpose(1, 2).reshape(max_seqlen_q, -1)
        return out

    def apply_fa4(
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
        # (L, H, D) -> (B, L, H, D)
        qt = q.unsqueeze(0).transpose(1, 2).contiguous()
        kt = k.unsqueeze(0).transpose(1, 2).contiguous()
        sparse_map, lut, real_topk = get_block_map(qt, kt, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)

        # (L, H, D) -> (B, L, H, D)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

        # (B, H, Q_block_num, K_block_num)
        full_block_idx, full_block_cnt = block_map_ordinal_lut_triton(sparse_map)
        mask_block_cnt = torch.zeros_like(full_block_cnt)
        mask_block_idx = torch.zeros_like(full_block_idx)

        out, _ = flash_attn_func_v4(
            q=q,
            k=k,
            v=v,
            mask_block_cnt=mask_block_cnt,
            mask_block_idx=mask_block_idx,
            full_block_cnt=full_block_cnt,
            full_block_idx=full_block_idx,
            block_size=(self.BLKQ, self.BLKK),
        )
        out = out.reshape(max_seqlen_q, -1)
        return out

    def apply_magi(
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
        # (L, H, D) -> (B, H, L, D)
        q_block_map, k_block_map = q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2)
        q_block_map = q_block_map.contiguous()
        k_block_map = k_block_map.contiguous()

        sparse_map, lut, real_topk = get_block_map(q_block_map, k_block_map, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)
        seqlen, head_num, head_dim = q.shape

        q_ranges, k_ranges = self.generate_qk_ranges(sparse_map[0], self.BLKQ, self.BLKK, seqlen)
        attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device="cpu").to(q.device, non_blocking=True)

        q = q.permute(1, 0, 2).reshape(head_num * seqlen, 1, head_dim)
        k = k.permute(1, 0, 2).reshape(head_num * seqlen, 1, head_dim)
        v = v.permute(1, 0, 2).reshape(head_num * seqlen, 1, head_dim)

        out = magi_ffa_func(
            q,
            k,
            v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            auto_range_merge=True,
        )[0]

        out = out.reshape(head_num, seqlen, head_dim).permute(1, 0, 2)

        return out.reshape(out.shape[0], -1)

    def generate_qk_ranges(self, mask, q_block_size, k_block_size, seqlen):
        # mask: [H, Q_block_num, K_block_num]
        h_indices, i_indices, j_indices = torch.nonzero(mask, as_tuple=True)

        base_offset = h_indices * seqlen

        q_start = base_offset + i_indices * q_block_size
        q_end = base_offset + torch.clamp((i_indices + 1) * q_block_size, max=seqlen)

        k_start = base_offset + j_indices * k_block_size
        k_end = base_offset + torch.clamp((j_indices + 1) * k_block_size, max=seqlen)

        q_ranges = torch.stack([q_start, q_end], dim=1).to(dtype=torch.int32)
        k_ranges = torch.stack([k_start, k_end], dim=1).to(dtype=torch.int32)

        return q_ranges, k_ranges
