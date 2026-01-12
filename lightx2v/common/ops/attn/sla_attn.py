import torch
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .kernels.sla_kernel import _attention
from .template import AttnWeightTemplate
from .utils.sla_util import get_block_map, get_cuda_arch

try:
    import spas_sage_attn._fused as fused
    import spas_sage_attn._qattn as qattn
    from spas_sage_attn.utils import block_map_lut_triton, get_vanilla_qk_quant
except ImportError:
    logger.warning("spas_sage_attn is not installed. SageSparseLinearAttention will not be available.")

SAGE2PP_ENABLED = True
try:
    from spas_sage_attn._qattn import qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
except ImportError:
    SAGE2PP_ENABLED = False

try:
    from magi_attention.functional import flex_flash_attn_func as magi_ffa_func
except ImportError:
    magi_ffa_func = None


@ATTN_WEIGHT_REGISTER("sla_attn")
class SlaAttnWeight(AttnWeightTemplate):
    sparsity_ratio = 0.8
    operator = "triton"

    def __init__(self):
        self.config = {}

        self.arch = get_cuda_arch(torch.cuda.current_device())
        self.topk = 1 - self.sparsity_ratio
        if self.operator == "triton":
            self.BLKQ, self.BLKK = 128, 128
            self.apply_func = self.apply_triton
        elif self.operator == "sage":
            if self.arch == "sm90":
                self.BLKQ, self.BLKK = 64, 128
            else:
                self.BLKQ, self.BLKK = 128, 64
            self.apply_func = self.apply_sage
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
        model_cls=None,
    ):
        return self.apply_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, model_cls)

    def apply_triton(
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

    def apply_sage(
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

        km = k.mean(dim=-2, keepdim=True)
        headdim = q.size(-1)

        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, self.BLKQ, self.BLKK)
        lut, valid_block_num = block_map_lut_triton(sparse_map)
        scale = 1.0 / (headdim**0.5)

        assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

        o_s = torch.empty_like(q)

        if self.arch in ("sm80", "sm86", "sm87"):
            pvthreshold = torch.full((q.shape[-3],), 1e6, dtype=torch.float32, device=q.device)
            v_fp16 = v.to(torch.float16)
            qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v_fp16, o_s, lut, valid_block_num, pvthreshold, q_scale, k_scale, 1, False, 1, scale, 0)
        else:
            b, h_kv, kv_len, head_dim = v.shape
            padded_len = (kv_len + 127) // 128 * 128
            v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
            fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
            v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
            v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
            fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

            if self.arch == "sm90":
                qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90(q_int8, k_int8, v_fp8, o_s, lut, valid_block_num, q_scale, k_scale, v_scale, 1, False, 1, scale)
            else:
                pvthreshold = torch.full((q.shape[-3],), 1e6, dtype=torch.float32, device=q.device)
                if SAGE2PP_ENABLED:
                    qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                        q_int8, k_int8, v_fp8, o_s, lut, valid_block_num, pvthreshold, q_scale, k_scale, v_scale, 1, False, 1, scale, 0
                    )
                else:
                    qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                        q_int8, k_int8, v_fp8, o_s, lut, valid_block_num, pvthreshold, q_scale, k_scale, v_scale, 1, False, 1, scale, 0
                    )

        out = o_s.transpose(1, 2).reshape(max_seqlen_q, -1)

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
        model_cls=None,
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
