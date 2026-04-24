import torch
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate
from .utils.sla_util import get_block_map, get_cuda_arch
from .utils.sparge_util import (
    block_map_incremental_lut_triton,
    block_map_ordinal_lut_triton,
    get_block_map_meansim,
    sage2_block_sparse_attn,
)

try:
    from sageattn3_sparse import sage3_block_sparse_attn
except ImportError:
    logger.info("sageattn3_sparse not found, please install sageattn3_sparse first")
    sage3_block_sparse_attn = None

capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
if capability in [(8, 9), (12, 0)]:
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
    from sageattn3_sparse import sparse_sageattn3
except ImportError:
    logger.info("sageattn3_sparse not found, please install sageattention sparse first")
    sparse_sageattn3 = None

try:
    from sageattention._qattn_sm90 import qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
    from sageattention.triton.quant_per_thread import quant_query_per_thread_int8_kernel
except ImportError:
    quant_query_per_thread_int8_kernel = None
    qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf = None
    logger.info("sageattention not found, please install sageattention first")


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
        **kwargs,
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
        **kwargs,
    ):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if len(q.shape) == 3:
            bs = 1
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif len(q.shape) == 4:
            bs = q.shape[0]

        x = sageattn3_blackwell(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).reshape(bs * max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("spas_sage_attn2")
class SparseSageAttn2Weight(AttnWeightTemplate):
    sparsity_ratio = 0.8
    sparse_mode = "sla_mode"

    def __init__(self):
        self.config = {}
        self.topk = 1 - self.sparsity_ratio

        self.arch = get_cuda_arch(torch.cuda.current_device())
        if self.arch == "sm90":
            self.BLKQ, self.BLKK = 64, 128
        else:
            self.BLKQ, self.BLKK = 128, 64

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
        # (L, H, D) -> (B, L, H, D)
        q = q.unsqueeze(0).transpose(1, 2).contiguous()
        k = k.unsqueeze(0).transpose(1, 2).contiguous()
        v = v.unsqueeze(0).transpose(1, 2).contiguous()
        bs = q.shape[0]

        if self.sparse_mode == "sla_mode":
            sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)
        elif self.sparse_mode == "sparge_mode":
            smooth_k = k - k.mean(dim=-2, keepdim=True)
            sparse_map = get_block_map_meansim(
                q,
                smooth_k,
                cdfthreshd=None,
                topk=self.topk,
                return_lut=False,
                BLKQ=self.BLKQ,
                BLKK=self.BLKK,
            )
        else:
            logger.info(f"spas_sage_attn2 sparse_mode only support sla_mode and sparge_mode now.")

        lut, valid_block_num = block_map_incremental_lut_triton(sparse_map)
        x = sage2_block_sparse_attn(q, k, v, lut, valid_block_num, self.BLKQ, self.BLKK, self.arch)
        x = x.transpose(1, 2).reshape(bs * max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("spas_sage_attn3")
class SparseSageAttn3Weight(AttnWeightTemplate):
    sparsity_ratio = 0.8
    sparse_mode = "sla_mode"
    per_block_mean = False

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
        # (L, H, D) -> (B, L, H, D)
        q = q.unsqueeze(0).transpose(1, 2).contiguous()
        k = k.unsqueeze(0).transpose(1, 2).contiguous()
        v = v.unsqueeze(0).transpose(1, 2).contiguous()
        bs = q.shape[0]

        if self.sparse_mode == "sla_mode":
            sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)
        elif self.sparse_mode == "sparge_mode":
            smooth_k = k - k.mean(dim=-2, keepdim=True)
            sparse_map = get_block_map_meansim(
                q,
                smooth_k,
                cdfthreshd=None,
                topk=self.topk,
                return_lut=False,
                BLKQ=self.BLKQ,
                BLKK=self.BLKK,
            )
        else:
            logger.info(f"spas_sage_attn3 sparse_mode only support sla_mode and sparge_mode now.")

        lut, valid_block_num = block_map_ordinal_lut_triton(sparse_map)
        x = sage3_block_sparse_attn(q, k, v, lut, valid_block_num, per_block_mean=self.per_block_mean)
        x = x.transpose(1, 2).reshape(bs * max_seqlen_q, -1)
        return x


@ATTN_WEIGHT_REGISTER("sage_attn2_k_int8_v_fp8")
class SageAttn2KInt8VFP8Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def quant_query_per_thread_int8(self, q, BLKQ=128, WARPQ=32, sm_scale=None):
        q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
        b, qo_len, h_qo, head_dim = q.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = (
            q_int8.stride(0),
            q_int8.stride(2),
            q_int8.stride(1),
        )
        q_scale = torch.empty(
            (b, h_qo, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8),
            device=q.device,
            dtype=torch.float32,
        )

        if sm_scale is None:
            sm_scale = head_dim**-0.5
        grid = ((qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, h_qo, b)
        quant_query_per_thread_int8_kernel[grid](
            q,
            q_int8,
            q_scale,
            qo_len,
            stride_bz_q,
            stride_h_q,
            stride_seq_q,
            stride_bz_qo,
            stride_h_qo,
            stride_seq_qo,
            q_scale.stride(0),
            q_scale.stride(1),
            C=head_dim,
            BLK=WARPQ,
        )
        return q_int8, q_scale

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        sm_scale=None,
        **kwargs,
    ):
        k_int8, k_scale = k
        v_fp8, v_scale = v
        q, k_int8, v_fp8 = q.contiguous(), k_int8.contiguous(), v_fp8.contiguous()
        assert torch.cuda.get_device_capability(q.device) == (9, 0)
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert k_int8.dtype == torch.int8
        assert k_scale is not None
        assert v_scale is not None
        assert q.stride(-1) == 1 and k_int8.stride(-1) == 1

        dtype = q.dtype

        if len(q.shape) == 3:
            bs = 1
            q = q.unsqueeze(0)
            if len(k_int8.shape) == 3:
                k_int8 = k_int8.unsqueeze(0)
            if len(v_fp8.shape) == 3:
                v_fp8 = v_fp8.unsqueeze(0)
        elif len(q.shape) == 4:
            bs = q.shape[0]

        head_dim_og = q.size(-1)
        if sm_scale is None:
            sm_scale = float(head_dim_og**-0.5)

        q_int8, q_scale = self.quant_query_per_thread_int8(q, BLKQ=64, WARPQ=16)
        o = torch.empty(q.size(), dtype=dtype, device=q.device)
        qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
            q_int8,
            k_int8,
            v_fp8,
            o,
            q_scale,
            k_scale,
            v_scale,
            0,
            0,
            3,
            sm_scale,
            0,
        )
        o = o.view(bs * max_seqlen_q, -1)
        return o
