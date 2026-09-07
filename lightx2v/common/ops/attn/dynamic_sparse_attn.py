import inspect
from functools import wraps

import torch
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .kernels.sla_kernel import _attention
from .kernels.sla_kernel_ar import _attention_ar
from .template import AttnWeightTemplate
from .utils.sla_util import block_lut_to_ordinal_metadata, get_block_map, get_cuda_arch
from .utils.sla_util_blhd import get_block_lut_blhd, get_block_map_blhd
from .utils.sparge_util import block_map_incremental_lut_triton, block_map_ordinal_lut_triton, sage2_block_sparse_attn

try:
    from flash_attn.cute import flash_attn_func as flash_attn_func_v4
except (ImportError, AttributeError) as exc:
    logger.info(f"FlashAttention 4 is unavailable: {exc}")
    flash_attn_func_v4 = None

try:
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
except (ImportError, AttributeError):
    BlockSparseTensorsTorch = None

try:
    from sageattn3_sparse import sage3_block_sparse_attn
except ImportError:
    logger.info("sageattn3_sparse not found, please install sageattn3_sparse first")
    sage3_block_sparse_attn = None

try:
    from magi_attention.functional import flex_flash_attn_func as magi_ffa_func
except ImportError:
    magi_ffa_func = None


def _detect_fa4_sparse_api():
    if flash_attn_func_v4 is None:
        return None
    try:
        parameters = inspect.signature(flash_attn_func_v4).parameters
    except (TypeError, ValueError):
        return None

    if "block_sparse_tensors" in parameters:
        return "block_sparse_tensors"

    expanded_parameters = {
        "mask_block_cnt",
        "mask_block_idx",
        "full_block_cnt",
        "full_block_idx",
        "block_size",
    }
    if expanded_parameters.issubset(parameters):
        return "expanded"
    return None


_FA4_SPARSE_API = _detect_fa4_sparse_api()


_FA4_BLOCKSPARSE_OP = None


if flash_attn_func_v4 is not None:
    _fa4_impl = flash_attn_func_v4

    # Keep the FA4 kernel eager while exposing a tensor-only boundary to
    # Dynamo; upstream blocksparse/callable FA4 is eager-only today.
    if hasattr(torch.library, "custom_op"):
        try:

            @torch.library.custom_op("lightx2v_internal::fa4_blocksparse", mutates_args=())
            def _fa4_blocksparse_op(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask_block_cnt: torch.Tensor,
                mask_block_idx: torch.Tensor,
                full_block_cnt: torch.Tensor,
                full_block_idx: torch.Tensor,
                block_q: int,
                block_k: int,
            ) -> torch.Tensor:
                sparse_kwargs = {
                    "mask_block_cnt": mask_block_cnt,
                    "mask_block_idx": mask_block_idx,
                    "full_block_cnt": full_block_cnt,
                    "full_block_idx": full_block_idx,
                    "block_size": (block_q, block_k),
                }
                if _FA4_SPARSE_API == "block_sparse_tensors":
                    sparse_kwargs = {"block_sparse_tensors": BlockSparseTensorsTorch(**sparse_kwargs)}
                out, _ = _fa4_impl(q=q, k=k, v=v, **sparse_kwargs)
                return out

            @_fa4_blocksparse_op.register_fake
            def _fa4_blocksparse_op_fake(
                q,
                k,
                v,
                mask_block_cnt,
                mask_block_idx,
                full_block_cnt,
                full_block_idx,
                block_q,
                block_k,
            ):
                return torch.empty_like(q)

            _FA4_BLOCKSPARSE_OP = _fa4_blocksparse_op
        except RuntimeError as exc:
            # Module reloads can encounter an already registered op.
            logger.debug("FA4 blocksparse custom op registration skipped: {}", exc)

    @torch.compiler.disable
    @wraps(_fa4_impl)
    def flash_attn_func_v4(*args, **kwargs):
        """Keep the blocksparse FA4 wrapper and kernel outside Dynamo fake mode."""
        return _fa4_impl(*args, **kwargs)


@torch.library.custom_op(
    "lightx2v::dynamic_sparse_sage2",
    mutates_args=(),
    device_types="cuda",
)
def dynamic_sparse_sage2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_ratio: float,
    block_q: int,
    block_k: int,
    arch: str,
) -> torch.Tensor:
    sparse_map, _, _ = get_block_map(q, k, topk_ratio=topk_ratio, BLKQ=block_q, BLKK=block_k)
    lut, valid_block_num = block_map_incremental_lut_triton(sparse_map)
    return sage2_block_sparse_attn(q, k, v, lut, valid_block_num, block_q, block_k, arch)


@dynamic_sparse_sage2.register_fake
def _dynamic_sparse_sage2_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_ratio: float,
    block_q: int,
    block_k: int,
    arch: str,
) -> torch.Tensor:
    return torch.empty_like(q)


@ATTN_WEIGHT_REGISTER("dynamic_sparse_attn")
class DynamicSparseAttnWeight(AttnWeightTemplate):
    sparsity_ratio = 0.8
    operator = "triton"
    per_block_mean = False

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.sparsity_ratio = float(self.config.get("sparsity_ratio", type(self).sparsity_ratio))
        self.operator = self.config.get("operator", type(self).operator)
        self.per_block_mean = bool(self.config.get("per_block_mean", type(self).per_block_mean))

        if not 0.0 <= self.sparsity_ratio < 1.0:
            raise ValueError(f"dynamic sparse attention sparsity_ratio must be in [0, 1), got {self.sparsity_ratio}")

        self.arch = get_cuda_arch(torch.cuda.current_device())
        self.topk = 1 - self.sparsity_ratio
        if self.operator == "triton":
            self.BLKQ, self.BLKK = 64, 64
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
        elif self.operator in ("cutedsl", "cutedsl_vit_fmha"):
            self.BLKQ, self.BLKK = 256, 128
            from .cutedsl_sparse_fmha import cutedsl_sparse_fmha

            self.cutedsl_sparse_fmha = cutedsl_sparse_fmha
            self.apply_func = self.apply_cutedsl
        elif self.operator == "cutedsl_fp8":
            # Low-precision Q/K/V on the same CuTeDSL block-sparse FMHA:
            # cast BF16 straight to FP8 E4M3 compute operands.
            self.BLKQ, self.BLKK = 256, 128
            from .cutedsl_sparse_fmha import cutedsl_sparse_fmha

            self.cutedsl_sparse_fmha = cutedsl_sparse_fmha
            self.apply_func = self.apply_cutedsl_fp8
        elif self.operator == "fa4":
            self.BLKQ, self.BLKK = 128, 128
            self.apply_func = self.apply_fa4
        elif self.operator == "magi":
            self.BLKQ, self.BLKK = 128, 128
            self.apply_func = self.apply_magi
        else:
            raise NotImplementedError(f"Not supported SLA operator: {self.operator}.")

        # logger.info(f"DynamicSparseAttnWeight: sparsity_ratio={self.sparsity_ratio}, operator={self.operator}, topk={self.topk}, BLKQ={self.BLKQ}, BLKK={self.BLKK}")

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
        if max_seqlen_q is None:
            max_seqlen_q = q.shape[0]
        if max_seqlen_kv is None:
            max_seqlen_kv = k.shape[0]
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
        # (L, H, D) -> (B, L, H, D)
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

        out = dynamic_sparse_sage2(q, k, v, self.topk, self.BLKQ, self.BLKK, self.arch)
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

    def apply_cutedsl(
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
        del cu_seqlens_kv, max_seqlen_kv, kwargs
        seqlen = q.shape[0]

        if q.shape[0] != k.shape[0] or k.shape != v.shape:
            raise ValueError("CuTeDSL sparse attention only supports self-attention")
        if cu_seqlens_q is not None and cu_seqlens_q.numel() != 2:
            raise ValueError("dynamic CuTeDSL block selection currently requires batch size 1")

        # Pool directly from the layout consumed by CuTeDSL so full Q/K stay
        # in their projection output buffers.
        lut, _, num_k_blocks = get_block_lut_blhd(
            q.unsqueeze(0),
            k.unsqueeze(0),
            topk_ratio=self.topk,
            BLKQ=self.BLKQ,
            BLKK=self.BLKK,
        )
        if lut.shape[-1] == 0:
            raise ValueError("CuTeDSL sparse attention requires at least one selected KV block")
        block_indices, block_count = block_lut_to_ordinal_metadata(lut, num_k_blocks)

        if cu_seqlens_q is None:
            cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=q.device)
        if max_seqlen_q is None:
            max_seqlen_q = seqlen

        out = self.cutedsl_sparse_fmha(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_q,
            max_seqlen_q,
            block_count,
            block_indices,
        )
        return out.reshape(seqlen, -1)

    def apply_cutedsl_fp8(
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
        del cu_seqlens_kv, max_seqlen_kv, kwargs
        seqlen = q.shape[0]

        if q.shape[0] != k.shape[0] or k.shape != v.shape:
            raise ValueError("CuTeDSL FP8 sparse attention only supports self-attention")
        if cu_seqlens_q is not None and cu_seqlens_q.numel() != 2:
            raise ValueError("dynamic CuTeDSL block selection currently requires batch size 1")

        # Select blocks on the full-precision Q/K before any quantization.
        lut, _, num_k_blocks = get_block_lut_blhd(
            q.unsqueeze(0),
            k.unsqueeze(0),
            topk_ratio=self.topk,
            BLKQ=self.BLKQ,
            BLKK=self.BLKK,
        )
        if lut.shape[-1] == 0:
            raise ValueError("CuTeDSL sparse attention requires at least one selected KV block")
        block_indices, block_count = block_lut_to_ordinal_metadata(lut, num_k_blocks)

        if cu_seqlens_q is None:
            cu_seqlens_q = torch.tensor([0, seqlen], dtype=torch.int32, device=q.device)
        if max_seqlen_q is None:
            max_seqlen_q = seqlen

        # Cast low-precision operands straight to FP8 compute tensors.
        q8 = q.to(torch.float8_e4m3fn).contiguous()
        k8 = k.to(torch.float8_e4m3fn).contiguous()
        v8 = v.to(torch.float8_e4m3fn).contiguous()

        out = self.cutedsl_sparse_fmha(
            q8,
            k8,
            v8,
            cu_seqlens_q,
            max_seqlen_q,
            block_count,
            block_indices,
            output_dtype=torch.bfloat16,
        )
        return out.reshape(seqlen, -1)

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
        # Build the sparse LUT directly from FA4's BLHD input layout. Pooling
        # only writes one vector per block instead of cloning full-size Q/K.
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        lut, _, num_k_blocks = get_block_lut_blhd(
            q,
            k,
            topk_ratio=self.topk,
            BLKQ=self.BLKQ,
            BLKK=self.BLKK,
        )

        # (B, H, Q_block_num, K_block_num)
        full_block_idx, full_block_cnt = block_lut_to_ordinal_metadata(lut, num_k_blocks)
        mask_block_cnt = torch.zeros_like(full_block_cnt)
        mask_block_idx = torch.zeros_like(full_block_idx)
        sparse_kwargs = {
            "mask_block_cnt": mask_block_cnt,
            "mask_block_idx": mask_block_idx,
            "full_block_cnt": full_block_cnt,
            "full_block_idx": full_block_idx,
            "block_size": (self.BLKQ, self.BLKK),
        }
        if _FA4_SPARSE_API == "block_sparse_tensors":
            if BlockSparseTensorsTorch is None:
                raise RuntimeError("FA4 expects block_sparse_tensors, but BlockSparseTensorsTorch is unavailable")
            if _FA4_BLOCKSPARSE_OP is not None and torch.compiler.is_compiling():
                out = _FA4_BLOCKSPARSE_OP(
                    q,
                    k,
                    v,
                    mask_block_cnt,
                    mask_block_idx,
                    full_block_cnt,
                    full_block_idx,
                    self.BLKQ,
                    self.BLKK,
                )
            else:
                out, _ = flash_attn_func_v4(
                    q=q,
                    k=k,
                    v=v,
                    block_sparse_tensors=BlockSparseTensorsTorch(**sparse_kwargs),
                )
        elif _FA4_SPARSE_API == "expanded":
            if _FA4_BLOCKSPARSE_OP is not None and torch.compiler.is_compiling():
                out = _FA4_BLOCKSPARSE_OP(
                    q,
                    k,
                    v,
                    mask_block_cnt,
                    mask_block_idx,
                    full_block_cnt,
                    full_block_idx,
                    self.BLKQ,
                    self.BLKK,
                )
            else:
                out, _ = flash_attn_func_v4(q=q, k=k, v=v, **sparse_kwargs)
        else:
            raise RuntimeError("Unsupported FA4 sparse attention API: expected block_sparse_tensors or expanded sparse parameters")
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
