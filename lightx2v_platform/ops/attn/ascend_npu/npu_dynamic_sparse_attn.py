import math
from functools import lru_cache

import torch
from loguru import logger

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

_V3_BACKEND = "mindie_rf_v3"
_V2_BACKEND = "mindie_rf_v2"
# The current CANN 8.5.1 image can import the MindIE RF v3 Python wrapper,
# but libopapi.so does not export the aclnnBlockSparseAttention operator it
# calls. Remember the first confirmed capability failure so later layers and
# model instances do not repeatedly enter the unsupported v3 path.
_V3_RUNTIME_UNAVAILABLE = False


def block_mean(x, block_size):
    """Mean-pool a BNSD tensor along S without padding the tail block."""
    if x.ndim != 4:
        raise ValueError(f"block_mean expects a BNSD tensor, but got shape {tuple(x.shape)}.")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, but got {block_size!r}.")

    sequence_length = x.shape[2]
    if sequence_length == 0:
        raise ValueError("block_mean does not support an empty sequence.")

    full_block_count, tail_size = divmod(sequence_length, block_size)
    pooled = []
    if full_block_count:
        full_blocks = x[:, :, : full_block_count * block_size, :]
        full_blocks = full_blocks.reshape(
            x.shape[0],
            x.shape[1],
            full_block_count,
            block_size,
            x.shape[3],
        )
        pooled.append(full_blocks.mean(dim=3))
    if tail_size:
        pooled.append(x[:, :, full_block_count * block_size :, :].mean(dim=2, keepdim=True))

    return pooled[0] if len(pooled) == 1 else torch.cat(pooled, dim=2)


def _validate_qk(q, k, sparsity_ratio, q_block_size, kv_block_size):
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(
            "build_dynamic_sparse_mask expects q and k in BNSD layout, "
            f"but got q={tuple(q.shape)}, k={tuple(k.shape)}."
        )
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"q and k batch sizes must match, but got {q.shape[0]} and {k.shape[0]}.")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q and k head dimensions must match, but got {q.shape[-1]} and {k.shape[-1]}.")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "The query head count must be divisible by the key/value head count for GQA, "
            f"but got {q.shape[1]} and {k.shape[1]}."
        )
    if q.device != k.device:
        raise ValueError(f"q and k must be on the same device, but got {q.device} and {k.device}.")
    if q.dtype != k.dtype:
        raise ValueError(f"q and k must have the same dtype, but got {q.dtype} and {k.dtype}.")
    if not q.is_floating_point() or not k.is_floating_point():
        raise TypeError(f"q and k must be floating-point tensors, but got {q.dtype} and {k.dtype}.")
    if q.shape[2] == 0 or k.shape[2] == 0:
        raise ValueError("q and k sequence lengths must be greater than zero.")
    if not isinstance(sparsity_ratio, (int, float)) or isinstance(sparsity_ratio, bool):
        raise TypeError(f"sparsity_ratio must be a number, but got {type(sparsity_ratio).__name__}.")
    if not 0 <= sparsity_ratio < 1:
        raise ValueError(f"sparsity_ratio must be in [0, 1), but got {sparsity_ratio}.")
    for name, size in (("q_block_size", q_block_size), ("kv_block_size", kv_block_size)):
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"{name} must be a positive integer, but got {size!r}.")


def build_dynamic_sparse_mask(
    q,
    k,
    sparsity_ratio,
    q_block_size=128,
    kv_block_size=128,
    return_topk_indices=False,
):
    """Build an exact-TopK int8 block mask from current BNSD query and key.

    K is smoothed over its sequence dimension before Q and K are mean-pooled.
    The returned tensor has shape ``[B, Hq, ceil(Sq/Bq), ceil(Sk/Bkv)]``.
    """
    _validate_qk(q, k, sparsity_ratio, q_block_size, kv_block_size)

    smoothed_k = k - k.mean(dim=2, keepdim=True)
    pooled_q = block_mean(q, q_block_size)
    pooled_k = block_mean(smoothed_k, kv_block_size)

    query_head_count = pooled_q.shape[1]
    key_value_head_count = pooled_k.shape[1]
    if query_head_count != key_value_head_count:
        pooled_k = pooled_k.repeat_interleave(query_head_count // key_value_head_count, dim=1)

    block_scores = torch.matmul(pooled_q, pooled_k.transpose(-1, -2))

    kv_block_count = block_scores.shape[-1]
    keep_block_count = math.ceil(kv_block_count * (1 - sparsity_ratio))
    keep_block_count = min(max(keep_block_count, 1), kv_block_count)
    topk_indices = torch.topk(
        block_scores,
        k=keep_block_count,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices

    block_mask = torch.zeros_like(block_scores, dtype=torch.int8)
    block_mask.scatter_(-1, topk_indices, 1)
    if return_topk_indices:
        return block_mask, topk_indices
    return block_mask


def topk_indices_to_v2_indices(topk_indices, kv_block_count):
    """Convert exact TopK indices to MindIE v2 selectors without a full-K sort."""
    if topk_indices.ndim != 4:
        raise ValueError(
            "topk_indices_to_v2_indices expects B,H,Q,Kkeep indices, "
            f"but got shape {tuple(topk_indices.shape)}."
        )
    if topk_indices.shape[0] != 1:
        raise ValueError(
            "MindIE RainFusion v2 selector conversion currently supports batch size 1, "
            f"but got {topk_indices.shape[0]}."
        )
    if not isinstance(kv_block_count, int) or isinstance(kv_block_count, bool) or kv_block_count <= 0:
        raise ValueError(f"kv_block_count must be a positive integer, but got {kv_block_count!r}.")
    keep_block_count = topk_indices.shape[-1]
    if keep_block_count == 0 or keep_block_count > kv_block_count:
        raise ValueError(
            "The number of TopK indices must be in [1, kv_block_count], "
            f"but got {keep_block_count} and kv_block_count={kv_block_count}."
        )

    sorted_indices = torch.sort(topk_indices.to(torch.int64), dim=-1).values
    padding_size = kv_block_count - keep_block_count
    if padding_size:
        padding = torch.full(
            (*sorted_indices.shape[:-1], padding_size),
            -1,
            dtype=torch.int64,
            device=sorted_indices.device,
        )
        sorted_indices = torch.cat((sorted_indices, padding), dim=-1)

    select_idx = sorted_indices[0].transpose(0, 1).contiguous()
    select_num_idx = torch.full(
        select_idx.shape[:2],
        keep_block_count,
        dtype=torch.int64,
        device=select_idx.device,
    )
    return select_idx, select_num_idx


def binary_mask_to_v2_indices(block_mask):
    """Convert a B=1 binary mask to the selector layout required by MindIE v2."""
    if block_mask.ndim != 4:
        raise ValueError(
            "binary_mask_to_v2_indices expects a BHQK mask, "
            f"but got shape {tuple(block_mask.shape)}."
        )
    if block_mask.shape[0] != 1:
        raise ValueError(
            "MindIE RainFusion v2 selector conversion currently supports batch size 1, "
            f"but got {block_mask.shape[0]}."
        )
    if block_mask.shape[-1] == 0:
        raise ValueError("The block mask must contain at least one KV block.")

    selected = block_mask.to(torch.bool)
    kv_block_count = selected.shape[-1]
    kv_indices = torch.arange(kv_block_count, dtype=torch.int64, device=selected.device)
    kv_indices = kv_indices.view(1, 1, 1, kv_block_count).expand_as(selected)
    padded_indices = torch.where(selected, kv_indices, kv_block_count)
    padded_indices = torch.sort(padded_indices, dim=-1).values
    padded_indices = torch.where(padded_indices == kv_block_count, -1, padded_indices)

    select_idx = padded_indices[0].transpose(0, 1).contiguous()
    select_num_idx = selected[0].transpose(0, 1).sum(dim=-1, dtype=torch.int64).contiguous()
    return select_idx, select_num_idx


def _import_v3_backend():
    from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import rain_fusion_attention_v3

    if not callable(rain_fusion_attention_v3):
        raise ImportError("mindiesd RainFusion v3 does not expose rain_fusion_attention_v3.")
    return rain_fusion_attention_v3


def _import_v2_backend():
    from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import rain_fusion_attention

    if not callable(rain_fusion_attention):
        raise ImportError("mindiesd RainFusion v2 does not expose rain_fusion_attention.")
    return rain_fusion_attention


def _is_v3_capability_error(error):
    """Match only the known CANN 8.5 missing-BlockSparseAttention failure."""
    message = str(error)
    return isinstance(error, RuntimeError) and "aclnnBlockSparseAttention" in message and "not in libopapi.so" in message


def _disable_v3_runtime(error):
    global _V3_RUNTIME_UNAVAILABLE

    if not _V3_RUNTIME_UNAVAILABLE:
        logger.warning(
            "MindIE RainFusion v3 cannot run because aclnnBlockSparseAttention "
            f"is unavailable; falling back to RainFusion v2. Original error: {error}"
        )
    _V3_RUNTIME_UNAVAILABLE = True
    # Backend resolution is cached. Clear it so the next layer observes the
    # process-wide capability result and resolves directly to RF v2.
    _load_mindie_backend.cache_clear()


@lru_cache(maxsize=4)
def _load_mindie_backend(backend=_V2_BACKEND, allow_v2_fallback=True):
    """Resolve MindIE sparse attention for the current CANN 8.5 runtime.

    RF v2 is the compatibility default because CANN 8.5.1 provides
    aclnnRainFusionAttention but not aclnnBlockSparseAttention. RF v3 remains
    available as an explicit opt-in for newer runtimes; it may fall back to v2
    only when its API or required ACLNN capability is unavailable.
    """
    if backend in ("auto", "v3", _V3_BACKEND):
        if _V3_RUNTIME_UNAVAILABLE:
            if not allow_v2_fallback:
                raise RuntimeError(
                    "MindIE RainFusion v3 was disabled because "
                    "aclnnBlockSparseAttention is unavailable and v2 fallback is disabled."
                )
            try:
                return _V2_BACKEND, _import_v2_backend()
            except ImportError as v2_error:
                raise RuntimeError(
                    "MindIE RainFusion v3 is unavailable at runtime and its v2 fallback "
                    "could not be imported."
                ) from v2_error
        try:
            return _V3_BACKEND, _import_v3_backend()
        except ImportError as v3_error:
            if not allow_v2_fallback:
                raise RuntimeError(
                    "MindIE RainFusion v3 is unavailable and v2 fallback is disabled."
                ) from v3_error
            logger.warning("MindIE RainFusion v3 API is unavailable; falling back to RainFusion v2.")
            try:
                return _V2_BACKEND, _import_v2_backend()
            except ImportError as v2_error:
                raise RuntimeError(
                    "Neither MindIE RainFusion v3 nor its v2 fallback is available. "
                    "Install a compatible MindIE-SD package."
                ) from v2_error
    if backend in ("v2", _V2_BACKEND):
        try:
            return _V2_BACKEND, _import_v2_backend()
        except ImportError as v2_error:
            raise RuntimeError(
                "MindIE RainFusion v2 is unavailable. Install a compatible MindIE-SD package."
            ) from v2_error
    raise ValueError(
        f"Unsupported NPU dynamic sparse attention backend {backend!r}; "
        f"expected 'auto', '{_V3_BACKEND}', or '{_V2_BACKEND}'."
    )


def _validate_inputs(q, k, v):
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            "npu_dynamic_sparse_attn expects q, k, and v in [S,H,D] layout, "
            f"but got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
        )
    if k.shape != v.shape:
        raise ValueError(f"k and v must have identical shapes, but got {tuple(k.shape)} and {tuple(v.shape)}.")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q and k/v head dimensions must match, but got {q.shape[-1]} and {k.shape[-1]}.")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "The query head count must be divisible by the key/value head count for GQA, "
            f"but got {q.shape[1]} and {k.shape[1]}."
        )
    if q.shape[0] == 0 or k.shape[0] == 0:
        raise ValueError("q and k/v sequence lengths must be greater than zero.")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"q, k, and v must be on the same device, but got {q.device}, {k.device}, and {v.device}.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"q, k, and v must have the same dtype, but got {q.dtype}, {k.dtype}, and {v.dtype}.")
    if not q.is_floating_point():
        raise TypeError(f"q, k, and v must be floating-point tensors, but got {q.dtype}.")


def _validate_cu_seqlens(cu_seqlens, sequence_length, name):
    if cu_seqlens is None:
        return
    if torch.is_tensor(cu_seqlens):
        if cu_seqlens.numel() != 2:
            raise ValueError(
                f"{name} must describe exactly one sequence and contain 2 elements, "
                f"but got {cu_seqlens.numel()}."
            )
        if cu_seqlens.device.type != "cpu":
            return
        values = cu_seqlens.detach().tolist()
    else:
        values = list(cu_seqlens)
    if values != [0, sequence_length]:
        raise ValueError(
            f"{name} must describe exactly one sequence [0, {sequence_length}], "
            f"but got {values}. Packed batches are not supported."
        )


@PLATFORM_ATTN_WEIGHT_REGISTER("npu_dynamic_sparse_attn")
class NpuDynamicSparseAttnWeight(AttnWeightTemplate):
    sparsity_ratio = 0.9
    q_block_size = 128
    kv_block_size = 128
    # Keep unqualified configs on RF v2: the target CANN 8.5.1 runtime lacks
    # the aclnnBlockSparseAttention symbol required by MindIE RF v3.
    backend = _V2_BACKEND
    allow_v2_fallback = True
    v3_inner_precise = 0
    v2_inner_precise = 0

    def __init__(self):
        self.config = {}
        cls = type(self)
        self.sparsity_ratio = cls.sparsity_ratio
        self.q_block_size = cls.q_block_size
        self.kv_block_size = cls.kv_block_size
        self.backend = cls.backend
        self.allow_v2_fallback = cls.allow_v2_fallback
        self.v3_inner_precise = cls.v3_inner_precise
        self.v2_inner_precise = cls.v2_inner_precise

    @classmethod
    def configure(cls, setting):
        setting = setting or {}
        q_block_size = setting.get("q_block_size", setting.get("block_size", cls.q_block_size))
        kv_block_size = setting.get("kv_block_size", setting.get("block_size", cls.kv_block_size))
        if q_block_size != 128 or kv_block_size != 128:
            raise ValueError(
                "MindIE NPU dynamic sparse attention currently requires "
                f"q_block_size=kv_block_size=128, but got {q_block_size} and {kv_block_size}."
            )

        sparsity_ratio = setting.get("sparsity_ratio", setting.get("sparsity", cls.sparsity_ratio))
        if not isinstance(sparsity_ratio, (int, float)) or isinstance(sparsity_ratio, bool):
            raise TypeError(f"sparsity_ratio must be a number, but got {type(sparsity_ratio).__name__}.")
        if not 0 <= sparsity_ratio < 1:
            raise ValueError(f"sparsity_ratio must be in [0, 1), but got {sparsity_ratio}.")

        backend = setting.get("backend", cls.backend)
        if backend not in ("auto", "v3", "v2", _V3_BACKEND, _V2_BACKEND):
            raise ValueError(f"Unsupported NPU dynamic sparse attention backend {backend!r}.")
        allow_v2_fallback = setting.get("allow_v2_fallback", cls.allow_v2_fallback)
        if not isinstance(allow_v2_fallback, bool):
            raise TypeError(f"allow_v2_fallback must be bool, but got {type(allow_v2_fallback).__name__}.")

        v3_inner_precise = setting.get("v3_inner_precise", cls.v3_inner_precise)
        v2_inner_precise = setting.get("v2_inner_precise", cls.v2_inner_precise)
        for name, value in (
            ("v3_inner_precise", v3_inner_precise),
            ("v2_inner_precise", v2_inner_precise),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, but got {type(value).__name__}.")
        cls.sparsity_ratio = sparsity_ratio
        cls.q_block_size = q_block_size
        cls.kv_block_size = kv_block_size
        cls.backend = backend
        cls.allow_v2_fallback = allow_v2_fallback
        cls.v3_inner_precise = v3_inner_precise
        cls.v2_inner_precise = v2_inner_precise

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
        _validate_inputs(q, k, v)
        query_length, query_head_count, head_dim = q.shape
        kv_length, key_value_head_count, _ = k.shape

        if max_seqlen_q is not None and max_seqlen_q != query_length:
            raise ValueError(f"max_seqlen_q must equal {query_length}, but got {max_seqlen_q}.")
        if max_seqlen_kv is not None and max_seqlen_kv != kv_length:
            raise ValueError(f"max_seqlen_kv must equal {kv_length}, but got {max_seqlen_kv}.")
        _validate_cu_seqlens(cu_seqlens_q, query_length, "cu_seqlens_q")
        _validate_cu_seqlens(cu_seqlens_kv, kv_length, "cu_seqlens_kv")
        if kwargs.get("attn_mask") is not None:
            raise NotImplementedError("npu_dynamic_sparse_attn does not support an additional attention mask.")
        if kwargs.get("causal", False):
            raise NotImplementedError("npu_dynamic_sparse_attn currently supports non-causal attention only.")
        if kwargs.get("dropout", 0.0) not in (None, 0, 0.0):
            raise NotImplementedError("npu_dynamic_sparse_attn does not support attention dropout.")

        softmax_scale = kwargs.get("softmax_scale")
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        q_bnsd = q.transpose(0, 1).unsqueeze(0).contiguous()
        k_bnsd = k.transpose(0, 1).unsqueeze(0).contiguous()
        v_bnsd = v.transpose(0, 1).unsqueeze(0).contiguous()
        backend_name, backend_fn = _load_mindie_backend(self.backend, self.allow_v2_fallback)
        block_mask, topk_indices = build_dynamic_sparse_mask(
            q_bnsd,
            k_bnsd,
            self.sparsity_ratio,
            q_block_size=self.q_block_size,
            kv_block_size=self.kv_block_size,
            return_topk_indices=True,
        )

        if backend_name == _V3_BACKEND:
            try:
                out = backend_fn(
                    q_bnsd,
                    k_bnsd,
                    v_bnsd,
                    block_sparse_mask=block_mask,
                    scale=softmax_scale,
                    head_num=query_head_count,
                    num_key_value_heads=key_value_head_count,
                    input_layout="BNSD",
                    actual_seq_lengths=[query_length],
                    actual_seq_lengths_kv=[kv_length],
                    sparse_size=self.q_block_size,
                    inner_precise=kwargs.get("v3_inner_precise", self.v3_inner_precise),
                )
            except RuntimeError as error:
                # CANN 8.5.1 discovers the missing BSA symbol only when the v3
                # operator is first executed. Fall back for that exact
                # capability error; propagate shape, OOM, numerical, and other
                # kernel failures unchanged so real defects are not hidden.
                if not self.allow_v2_fallback or not _is_v3_capability_error(error):
                    raise
                _disable_v3_runtime(error)
                backend_name, backend_fn = _load_mindie_backend(self.backend, self.allow_v2_fallback)

        if backend_name == _V2_BACKEND:
            if query_head_count != key_value_head_count:
                raise NotImplementedError(
                    "MindIE RainFusion v2 fallback does not support GQA in this integration; "
                    f"got {query_head_count} query heads and {key_value_head_count} KV heads."
                )
            # RF v2 consumes padded block selectors instead of the binary
            # block mask accepted by RF v3. The Top-K choice itself is shared.
            select_idx, select_num_idx = topk_indices_to_v2_indices(
                topk_indices,
                kv_block_count=block_mask.shape[-1],
            )
            out = backend_fn(
                q_bnsd,
                k_bnsd,
                v_bnsd,
                scale=softmax_scale,
                head_num=query_head_count,
                input_layout="BNSD",
                select_idx=select_idx,
                select_num_idx=select_num_idx,
                blockshape=[self.q_block_size, self.kv_block_size],
                actual_seq_lengths=[query_length],
                actual_seq_lengths_kv=[kv_length],
                inner_precise=kwargs.get("v2_inner_precise", self.v2_inner_precise),
            )
        elif backend_name != _V3_BACKEND:
            raise RuntimeError(f"Internal error: unrecognized MindIE backend {backend_name!r}.")

        if not torch.is_tensor(out):
            raise TypeError(f"MindIE sparse attention must return a tensor, but got {type(out).__name__}.")
        expected_shape = (1, query_head_count, query_length, head_dim)
        if tuple(out.shape) != expected_shape:
            raise RuntimeError(
                f"MindIE sparse attention returned shape {tuple(out.shape)}, expected {expected_shape}."
            )
        return out.squeeze(0).transpose(0, 1).contiguous().reshape(query_length, query_head_count * head_dim)
