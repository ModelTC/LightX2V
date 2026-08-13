import torch
from loguru import logger

from .utils.sla_util import get_block_map
from .utils.sparge_util import block_map_ordinal_lut_triton, get_block_map_meansim

try:
    from flash_attn import flash_attn_func as flash_attn_func_v2
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v2
except ImportError:
    logger.info("flash_attn2 not found, please install flash_attn2 first")
    flash_attn_func_v2 = None
    flash_attn_varlen_func_v2 = None

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    logger.info("flash_attn3 not found, please install flash_attn3 first")
    flash_attn_func_v3 = None
    flash_attn_varlen_func_v3 = None

try:
    from flash_attn.cute import flash_attn_func as flash_attn_func_v4
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
except ImportError:
    logger.info("flash_attn.cute not found, please install flashattention4 first")
    flash_attn_func_v4 = None
    BlockSparseTensorsTorch = None


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
        causal = kwargs.get("causal", False)
        softmax_scale = kwargs.get("softmax_scale", None)
        if len(q.shape) == 3:
            bs = 1
        elif len(q.shape) == 4:
            bs = q.shape[0]
        total_seqlen = bs * max_seqlen_q

        if bs == 1:
            if len(q.shape) == 3:
                q = q.unsqueeze(0)
                k = k.unsqueeze(0)
                v = v.unsqueeze(0)
            x = flash_attn_func_v2(q, k, v, softmax_scale=softmax_scale, causal=causal).reshape(total_seqlen, -1)
        else:
            if cu_seqlens_q.is_cpu:
                cu_seqlens_q = cu_seqlens_q.to(q.device, non_blocking=True)
            if cu_seqlens_kv.is_cpu:
                cu_seqlens_kv = cu_seqlens_kv.to(k.device, non_blocking=True)
            if len(q.shape) == 4:
                q = q.reshape(-1, q.shape[-2], q.shape[-1])
                k = k.reshape(-1, k.shape[-2], k.shape[-1])
                v = v.reshape(-1, v.shape[-2], v.shape[-1])
            x = flash_attn_varlen_func_v2(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                softmax_scale=softmax_scale,
                causal=causal,
            ).reshape(total_seqlen, -1)

        return x

    def apply_with_lse(self, q, k, v, softmax_scale=None):
        """Apply one dense attention block and return LSE as [tokens, heads]."""
        if flash_attn_func_v2 is None:
            raise ImportError("FlashAttention2 is not installed.")
        if q.ndim == 3:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif q.ndim != 4:
            raise ValueError(f"Dense FlashAttention2 expects 3D or 4D Q/K/V, got q.ndim={q.ndim}.")

        output, lse, *_ = flash_attn_func_v2(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            softmax_scale=softmax_scale,
            causal=False,
            return_attn_probs=True,
        )
        output = output.reshape(q.shape[0] * q.shape[1], -1)
        lse = lse.transpose(1, 2).reshape(q.shape[0] * q.shape[1], q.shape[2])
        return output, lse


@ATTN_WEIGHT_REGISTER("flash_attn2_varlen")
class FlashAttn2VarlenWeight(AttnWeightTemplate):
    """FlashAttention-2 with the varlen kernel even for a dense batch.

    ``flash_attn2`` intentionally keeps its established single-batch dense
    fast path.  Some reference implementations, including Wan-Animate-2's
    CLIP encoder, always use ``flash_attn_varlen_func``; the two kernels are
    numerically close but not bit-identical.  This opt-in backend preserves the
    reference kernel choice without changing existing users.
    """

    def __init__(self):
        self.config = {}

    @staticmethod
    def _cu_seqlens(batch_size, seq_len, device):
        lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        return torch.cat((lengths.new_zeros(1), lengths)).cumsum(0, dtype=torch.int32)

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
        if flash_attn_varlen_func_v2 is None:
            raise ImportError("flash_attn2_varlen requires flash-attn v2.")
        if q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
            raise ValueError(f"flash_attn2_varlen expects matching 3D or 4D Q/K/V, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}.")

        if q.ndim == 4:
            batch_size, query_len = q.shape[:2]
            key_len = k.shape[1]
            total_query_len = batch_size * query_len
            q = q.flatten(0, 1)
            k = k.flatten(0, 1)
            v = v.flatten(0, 1)
        else:
            total_query_len = q.shape[0]
            batch_size = 1 if cu_seqlens_q is None else cu_seqlens_q.numel() - 1
            query_len = total_query_len
            key_len = k.shape[0]

        if cu_seqlens_q is None:
            cu_seqlens_q = self._cu_seqlens(batch_size, query_len, q.device)
        elif cu_seqlens_q.is_cpu:
            cu_seqlens_q = cu_seqlens_q.to(q.device, non_blocking=True)
        if cu_seqlens_kv is None:
            cu_seqlens_kv = self._cu_seqlens(batch_size, key_len, k.device)
        elif cu_seqlens_kv.is_cpu:
            cu_seqlens_kv = cu_seqlens_kv.to(k.device, non_blocking=True)

        if max_seqlen_q is None:
            max_seqlen_q = query_len if batch_size == 1 else int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
        if max_seqlen_kv is None:
            max_seqlen_kv = key_len if batch_size == 1 else int((cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).max().item())
        dropout_p = kwargs.get("dropout_p", kwargs.get("drop_rate", 0.0))
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        if kwargs.get("q_scale") is not None:
            q = q * kwargs["q_scale"]

        output = flash_attn_varlen_func_v2(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_kv,
            dropout_p=dropout_p,
            softmax_scale=kwargs.get("softmax_scale"),
            causal=kwargs.get("causal", False),
            window_size=kwargs.get("window_size", (-1, -1)),
            deterministic=kwargs.get("deterministic", False),
        )
        return output.reshape(total_query_len, -1)


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
        causal = kwargs.get("causal", False)
        softmax_scale = kwargs.get("softmax_scale", None)
        if len(q.shape) == 3:
            bs = 1
        elif len(q.shape) == 4:
            bs = q.shape[0]
        total_seqlen = bs * max_seqlen_q

        if bs == 1:
            if len(q.shape) == 3:
                q = q.unsqueeze(0)
                k = k.unsqueeze(0)
                v = v.unsqueeze(0)
            x = flash_attn_func_v3(q, k, v, softmax_scale=softmax_scale, causal=causal).reshape(total_seqlen, -1)
        else:
            if cu_seqlens_q.is_cpu:
                cu_seqlens_q = cu_seqlens_q.to(q.device, non_blocking=True)
            if cu_seqlens_kv.is_cpu:
                cu_seqlens_kv = cu_seqlens_kv.to(k.device, non_blocking=True)
            if len(q.shape) == 4:
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
                softmax_scale=softmax_scale,
                causal=causal,
            ).reshape(total_seqlen, -1)

        return x

    def apply_with_lse(self, q, k, v, softmax_scale=None):
        """Apply one dense attention block and return LSE as [tokens, heads]."""
        if flash_attn_func_v3 is None:
            raise ImportError("FlashAttention3 is not installed.")
        if q.ndim == 3:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        elif q.ndim != 4:
            raise ValueError(f"Dense FlashAttention3 expects 3D or 4D Q/K/V, got q.ndim={q.ndim}.")

        output, lse, *_ = flash_attn_func_v3(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            softmax_scale=softmax_scale,
            causal=False,
            return_attn_probs=True,
        )
        output = output.reshape(q.shape[0] * q.shape[1], -1)
        lse = lse.transpose(1, 2).reshape(q.shape[0] * q.shape[1], q.shape[2])
        return output, lse


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
        block_sparse_tensors = BlockSparseTensorsTorch(
            mask_block_cnt=mask_block_cnt,
            mask_block_idx=mask_block_idx,
            full_block_cnt=full_block_cnt,
            full_block_idx=full_block_idx,
            block_size=(self.BLKQ, self.BLKK),
        )

        x, _ = flash_attn_func_v4(
            q=q,
            k=k,
            v=v,
            block_sparse_tensors=block_sparse_tensors,
        )

        x = x.reshape(bs * max_seqlen_q, -1)
        return x
