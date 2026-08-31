from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from loguru import logger

from lightx2v_platform.registry_factory import PLATFORM_SPARSE_OPERATOR_REGISTER

try:
    from mate.jit.msa_ops import (
        get_msa_fwd_int8_block_fp8_bf16out_module,
        get_msa_fwd_int8_global_fp8_bf16out_module,
        get_msa_fwd_mixed_module,
    )
    from mate.msa_interface import MsaRuntimeMetadata, sparse_msa, sparse_msa_plan
except (ImportError, AttributeError) as exc:
    logger.info(f"MATE sparse MSA is unavailable: {exc}")
    MsaRuntimeMetadata = None
    sparse_msa = None
    sparse_msa_plan = None
    get_msa_fwd_mixed_module = None
    get_msa_fwd_int8_global_fp8_bf16out_module = None
    get_msa_fwd_int8_block_fp8_bf16out_module = None


_PLAN_CACHE = {}
_KV_INDICES_CACHE = {}
_RUNTIME_METADATA_CACHE = {}
_LSE_CACHE = {}
_VALIDATED_LUT_CONTRACTS = set()
_KERNEL_TOPK = 16
_FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)
_FP8_QUANTIZER = None
_FP8_V_QUANTIZER = None
_INT8_QK_FP8V_QUANTIZER = None
_INT8_BLOCK_QK_FP8V_QUANTIZERS = {}
_MIXED_FWD_KERNELS = {}
_INT8_FWD_KERNELS = {}
_INT8_BLOCK_FWD_KERNELS = {}


def _quantize_fp8_qkv(q, k, v, q_scale, k_scale, v_scale):
    global _FP8_QUANTIZER
    if _FP8_E4M3 is None:
        raise RuntimeError("torch.float8_e4m3fn is unavailable")
    if _FP8_QUANTIZER is None:

        def quantize(q_, k_, v_, q_scale_, k_scale_, v_scale_):
            fp8_max = 448.0
            return (
                torch.clamp(q_.float() / q_scale_, -fp8_max, fp8_max).to(_FP8_E4M3),
                torch.clamp(k_.float() / k_scale_, -fp8_max, fp8_max).to(_FP8_E4M3),
                torch.clamp(v_.float() / v_scale_, -fp8_max, fp8_max).to(_FP8_E4M3),
            )

        _FP8_QUANTIZER = torch.compile(quantize, dynamic=None)
    return _FP8_QUANTIZER(q, k, v, q_scale, k_scale, v_scale)


def _quantize_fp8_v(v, v_scale):
    global _FP8_V_QUANTIZER
    if _FP8_E4M3 is None:
        raise RuntimeError("torch.float8_e4m3fn is unavailable")
    if _FP8_V_QUANTIZER is None:

        def quantize(v_, v_scale_):
            return torch.clamp(v_.float() / v_scale_, -448.0, 448.0).to(_FP8_E4M3)

        _FP8_V_QUANTIZER = torch.compile(quantize, dynamic=None)
    return _FP8_V_QUANTIZER(v, v_scale)


def _quantize_int8_qk_fp8_v(q, k, v, q_scale, k_scale, v_scale):
    global _INT8_QK_FP8V_QUANTIZER
    if _FP8_E4M3 is None:
        raise RuntimeError("torch.float8_e4m3fn is unavailable")
    if _INT8_QK_FP8V_QUANTIZER is None:

        def quantize(q_, k_, v_, q_scale_, k_scale_, v_scale_):
            return (
                torch.clamp(torch.round(q_.float() / q_scale_), -127, 127).to(torch.int8),
                torch.clamp(torch.round(k_.float() / k_scale_), -127, 127).to(torch.int8),
                torch.clamp(v_.float() / v_scale_, -448.0, 448.0).to(_FP8_E4M3),
            )

        _INT8_QK_FP8V_QUANTIZER = torch.compile(quantize, dynamic=None)
    return _INT8_QK_FP8V_QUANTIZER(q, k, v, q_scale, k_scale, v_scale)


def _quantize_int8_block_qk_fp8_v(q, k, v, q_block_size, kv_len, v_scale):
    quantizer = _INT8_BLOCK_QK_FP8V_QUANTIZERS.get(q_block_size)
    if quantizer is None:

        def quantize(q_, k_, v_, kv_len_, v_scale_):
            q_padded = F.pad(q_, (0, 0, 0, 0, 0, (-q_.shape[0]) % q_block_size))
            q_blocks = q_padded.reshape(-1, q_block_size, q_.shape[1], q_.shape[2]).float()
            q_scale = q_blocks.abs().amax(dim=(1, 3)).clamp_min(1e-6) / 127.0
            q8 = torch.clamp(torch.round(q_blocks / q_scale[:, None, :, None]), -127, 127).to(torch.int8).reshape(-1, q_.shape[1], q_.shape[2])[: q_.shape[0]]

            k_float = k_.float()
            k_mean = k_float.reshape(-1, k_.shape[2], k_.shape[3])[:kv_len_].mean(dim=0)
            k_centered = k_float - k_mean[None, None, :, :]
            k_scale = k_centered.abs().amax(dim=(1, 3)).clamp_min(1e-6) / 127.0
            k8 = torch.clamp(torch.round(k_centered / k_scale[:, None, :, None]), -127, 127).to(torch.int8)
            v8 = torch.clamp(v_.float() / v_scale_, -448.0, 448.0).to(_FP8_E4M3)
            return q8, k8, v8, q_scale, k_scale

        quantizer = torch.compile(quantize, dynamic=None)
        _INT8_BLOCK_QK_FP8V_QUANTIZERS[q_block_size] = quantizer
    return quantizer(q, k, v, kv_len, v_scale)


def _get_mixed_fwd_kernel(topk, q_block_size):
    key = (topk, q_block_size)
    kernel = _MIXED_FWD_KERNELS.get(key)
    if kernel is None:
        if get_msa_fwd_mixed_module is None:
            raise RuntimeError("the installed MATE build has no mixed-PV MSA kernel")
        name = f"bf16_fp8pv_bf16out_qb{q_block_size}_topk{topk}"
        kernel = get_msa_fwd_mixed_module(topk, q_block_size).get_function(f"msa_fwd_{name}_causal_0")
        _MIXED_FWD_KERNELS[key] = kernel
    return kernel


def _get_int8_fwd_kernel(topk, q_block_size):
    key = (topk, q_block_size)
    kernel = _INT8_FWD_KERNELS.get(key)
    if kernel is None:
        if get_msa_fwd_int8_global_fp8_bf16out_module is None:
            raise RuntimeError("the installed MATE build has no INT8-QK sparse MSA kernel")
        name = f"int8qk_global_fp8pv_bf16out_qb{q_block_size}_topk{topk}"
        kernel = get_msa_fwd_int8_global_fp8_bf16out_module(topk, q_block_size).get_function(f"msa_fwd_{name}_causal_0")
        _INT8_FWD_KERNELS[key] = kernel
    return kernel


def _get_int8_block_fwd_kernel(topk, q_block_size):
    key = (topk, q_block_size)
    kernel = _INT8_BLOCK_FWD_KERNELS.get(key)
    if kernel is None:
        if get_msa_fwd_int8_block_fp8_bf16out_module is None:
            raise RuntimeError("the installed MATE build has no block-scaled INT8-QK sparse MSA kernel")
        name = f"int8qk_blockfused_fp8pv_bf16out_qb{q_block_size}_topk{topk}"
        kernel = get_msa_fwd_int8_block_fp8_bf16out_module(topk, q_block_size).get_function(f"msa_fwd_{name}_causal_0")
        _INT8_BLOCK_FWD_KERNELS[key] = kernel
    return kernel


def _get_plan(q_len, kv_len, num_q_heads, num_kv_heads, topk, causal):
    key = (q_len, kv_len, num_q_heads, num_kv_heads, topk, causal)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        q_lens = torch.tensor([q_len], dtype=torch.int32, device="cpu")
        kv_lens = torch.tensor([kv_len], dtype=torch.int32, device="cpu")
        plan = sparse_msa_plan(
            q_lens,
            kv_lens,
            num_q_heads,
            num_kv_heads,
            num_kv_splits=1,
            page_size=128,
            sparse_block_size=128,
            kv_block_num=topk,
            causal=causal,
        )
        _PLAN_CACHE[key] = plan
    return plan


def _get_kv_indices(q, num_pages):
    key = (q.device, num_pages)
    kv_indices = _KV_INDICES_CACHE.get(key)
    if kv_indices is None:
        kv_indices = torch.arange(num_pages, dtype=torch.int32, device=q.device)
        _KV_INDICES_CACHE[key] = kv_indices
    return kv_indices


def _get_runtime_metadata(q, q_len, kv_len, kv_indices):
    key = (q.device, q_len, kv_len)
    metadata = _RUNTIME_METADATA_CACHE.get(key)
    if metadata is None:
        metadata = MsaRuntimeMetadata(
            qo_lens=torch.tensor([q_len], dtype=torch.int32, device=q.device),
            kv_lens=torch.tensor([kv_len], dtype=torch.int32, device=q.device),
            qo_offset=torch.tensor([kv_len - q_len], dtype=torch.int32, device=q.device),
            cu_seqlens_q=torch.tensor([0, q_len], dtype=torch.int32, device=q.device),
            cu_seqlens_k=torch.tensor([0, kv_len], dtype=torch.int32, device=q.device),
            kv_page_indptr=torch.tensor([0, kv_indices.numel()], dtype=torch.int32, device=q.device),
            seqused_k=torch.tensor([kv_len], dtype=torch.int32, device=q.device),
        )
        _RUNTIME_METADATA_CACHE[key] = metadata
    return metadata


def _get_lse_buffer(q, q_len, num_q_heads):
    stream = torch.musa.current_stream(q.device)
    key = (q.device, stream.musa_stream, q_len, num_q_heads)
    lse = _LSE_CACHE.get(key)
    if lse is None:
        lse = torch.empty(
            (q_len, num_q_heads),
            dtype=torch.float32,
            device=q.device,
        )
        _LSE_CACHE[key] = lse
    return lse


def _validate_lut_values(block_indexes, topk, num_pages):
    contract = (tuple(block_indexes.shape), topk, num_pages)
    if contract in _VALIDATED_LUT_CONTRACTS:
        return

    min_index = int(block_indexes.min().item())
    max_index = int(block_indexes.max().item())
    if min_index < 0 or max_index >= num_pages:
        raise ValueError(f"block indices must be in [0, {num_pages}), got [{min_index}, {max_index}]")
    sorted_indices = torch.sort(block_indexes, dim=-1).values
    if not bool(torch.all(sorted_indices[..., 1:] != sorted_indices[..., :-1]).item()):
        raise ValueError("each sparse LUT row must contain distinct block indices")

    _VALIDATED_LUT_CONTRACTS.add(contract)


@torch.library.custom_op(
    "lightx2v::musa_sparse_msa",
    mutates_args=(),
    device_types="cuda",
)
def musa_sparse_msa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indexes: torch.Tensor,
    q_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    topk: int,
    q_block_size: int,
    causal: bool,
    softmax_scale: float | None,
    validate_lut: bool,
    compute_mode: int,
    q_scale: float,
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    output_dtype = q.dtype
    num_pages = int(k.shape[0])
    if validate_lut:
        _validate_lut_values(block_indexes, topk, num_pages)
    kv_indices = _get_kv_indices(q, num_pages)
    runtime_metadata = _get_runtime_metadata(q, q_len, kv_len, kv_indices)
    if compute_mode == 1:
        q, k, v = _quantize_fp8_qkv(q, k, v, q_scale, k_scale, v_scale)
        base_scale = softmax_scale if softmax_scale is not None else 128**-0.5
        softmax_scale = base_scale * q_scale
    elif compute_mode == 0:
        k_scale = 1.0
        v_scale = 1.0

    if compute_mode == 2:
        v = _quantize_fp8_v(v, v_scale)
        out = torch.empty_like(q)
        lse = _get_lse_buffer(q, q_len, num_q_heads)
        _get_mixed_fwd_kernel(topk, q_block_size)(
            q,
            k,
            v,
            block_indexes,
            runtime_metadata.cu_seqlens_q,
            runtime_metadata.seqused_k,
            runtime_metadata.qo_offset,
            kv_indices,
            runtime_metadata.kv_page_indptr,
            q_len,
            kv_len,
            float(softmax_scale if softmax_scale is not None else 128**-0.5),
            1.0,
            v_scale,
            False,
            out,
            lse,
        )
        return out

    if compute_mode == 3:
        q8, k8, v8 = _quantize_int8_qk_fp8_v(q, k, v, q_scale, k_scale, v_scale)
        out = torch.empty_like(q)
        lse = _get_lse_buffer(q, q_len, num_q_heads)
        base_scale = softmax_scale if softmax_scale is not None else 128**-0.5
        _get_int8_fwd_kernel(topk, q_block_size)(
            q8,
            k8,
            v8,
            block_indexes,
            runtime_metadata.cu_seqlens_q,
            runtime_metadata.seqused_k,
            runtime_metadata.qo_offset,
            kv_indices,
            runtime_metadata.kv_page_indptr,
            q_len,
            kv_len,
            float(base_scale * q_scale),
            k_scale,
            v_scale,
            False,
            out,
            lse,
        )
        return out

    if compute_mode == 4:
        q8, k8, v8, q_scale_blocks, k_scale_blocks = _quantize_int8_block_qk_fp8_v(q, k, v, q_block_size, kv_len, v_scale)
        out = torch.empty_like(q)
        lse = _get_lse_buffer(q, q_len, num_q_heads)
        base_scale = softmax_scale if softmax_scale is not None else 128**-0.5
        _get_int8_block_fwd_kernel(topk, q_block_size)(
            q8,
            k8,
            v8,
            q_scale_blocks,
            k_scale_blocks,
            block_indexes,
            runtime_metadata.cu_seqlens_q,
            runtime_metadata.seqused_k,
            runtime_metadata.qo_offset,
            kv_indices,
            runtime_metadata.kv_page_indptr,
            q_len,
            kv_len,
            float(base_scale),
            1.0,
            v_scale,
            False,
            out,
            lse,
        )
        return out

    plan = _get_plan(
        q_len,
        kv_len,
        num_q_heads,
        num_kv_heads,
        _KERNEL_TOPK,
        causal,
    )
    if topk == _KERNEL_TOPK:
        lse = _get_lse_buffer(q, q_len, num_q_heads)
        out, _ = sparse_msa(
            q,
            k,
            v,
            plan,
            kv_indices=kv_indices,
            kv_block_indexes=block_indexes,
            sm_scale=softmax_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            lse=lse,
            runtime_metadata=runtime_metadata,
        )
        return out.to(output_dtype) if out.dtype != output_dtype else out

    partial_outputs = []
    partial_lses = []
    for begin in range(0, topk, _KERNEL_TOPK):
        end = min(begin + _KERNEL_TOPK, topk)
        chunk = block_indexes[..., begin:end]
        if end - begin < _KERNEL_TOPK:
            chunk = F.pad(chunk, (0, _KERNEL_TOPK - (end - begin)), value=-1)
        lse = torch.empty(
            (q_len, num_q_heads),
            dtype=torch.float32,
            device=q.device,
        )
        out, _ = sparse_msa(
            q,
            k,
            v,
            plan,
            kv_indices=kv_indices,
            kv_block_indexes=chunk.contiguous(),
            sm_scale=softmax_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            lse=lse,
            runtime_metadata=runtime_metadata,
        )
        partial_outputs.append(out.float())
        partial_lses.append(lse)

    stacked_lse = torch.stack(partial_lses, dim=0)
    weights = torch.softmax(stacked_lse, dim=0)
    stacked_out = torch.stack(partial_outputs, dim=0)
    return (stacked_out * weights[..., None]).sum(dim=0).to(dtype=output_dtype)


@musa_sparse_msa.register_fake
def _musa_sparse_msa_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indexes: torch.Tensor,
    q_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    topk: int,
    q_block_size: int,
    causal: bool,
    softmax_scale: float | None,
    validate_lut: bool,
    compute_mode: int,
    q_scale: float,
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    return torch.empty_like(q)


@PLATFORM_SPARSE_OPERATOR_REGISTER("musa_sparse")
class MusaSparseOperator:
    """Run LightX2V block-sparse attention with the MATE MSA kernel."""

    q_block_size = 64
    k_block_size = 128
    _SUPPORTED_TOPK = (8, 10, 11, 12, 16, 24, 32)
    _SUPPORTED_HEAD_RATIOS = (1, 8, 16)
    _native_mixed_topks = (8, 10, 11, 12, 16, 24, 32)

    def __init__(self, operator_setting=None):
        self.operator_setting = dict(operator_setting or {})
        self.topk = int(self.operator_setting.get("topk", 16))
        self.validate_lut = bool(
            self.operator_setting.get(
                "validate_lut",
                self.operator_setting.get("validate_mask", True),
            )
        )
        self.fallback_operator = self.operator_setting.get("fallback_operator")
        self.fallback_setting = dict(self.operator_setting.get("fallback_setting", {}))
        self._fallback = None
        self._warned_fallback_reasons = set()
        self.block_indices_only = self.fallback_operator is None
        self.center_k = bool(self.operator_setting.get("center_k", True))
        self.compute_dtype = self.operator_setting.get("compute_dtype", "mixed_bf16_fp8pv")
        if self.compute_dtype not in (
            "bf16",
            "fp8_e4m3",
            "mixed_bf16_fp8pv",
            "int8_qk_fp8pv",
            "int8_qk_block_fp8pv",
        ):
            raise ValueError(f"musa_sparse compute_dtype must be 'bf16', 'fp8_e4m3', 'mixed_bf16_fp8pv', 'int8_qk_fp8pv', or 'int8_qk_block_fp8pv', got {self.compute_dtype!r}")
        self.q_scale = float(self.operator_setting.get("q_scale", 1.0))
        self.k_scale = float(self.operator_setting.get("k_scale", 1.0))
        self.v_scale = float(self.operator_setting.get("v_scale", 1.0))
        if self.compute_dtype == "fp8_e4m3" and min(self.q_scale, self.k_scale, self.v_scale) <= 0.0:
            raise ValueError("musa_sparse FP8 scales must be positive")
        if self.compute_dtype == "mixed_bf16_fp8pv" and self.v_scale <= 0.0:
            raise ValueError("musa_sparse mixed-PV v_scale must be positive")
        if self.compute_dtype == "int8_qk_fp8pv" and min(self.q_scale, self.k_scale, self.v_scale) <= 0.0:
            raise ValueError("musa_sparse INT8-QK/FP8-PV scales must be positive")
        if self.compute_dtype == "int8_qk_block_fp8pv" and self.v_scale <= 0.0:
            raise ValueError("musa_sparse block-INT8-QK/FP8-PV v_scale must be positive")

        if self.topk not in self._SUPPORTED_TOPK:
            raise ValueError(f"musa_sparse topk must be one of {self._SUPPORTED_TOPK}, got {self.topk}")
        if self.fallback_operator == "musa_sparse":
            raise ValueError("musa_sparse cannot fall back to itself")

    def _get_fallback(self):
        if self.fallback_operator is None:
            return None
        if self._fallback is None:
            # Import lazily: the framework registry snapshots platform
            # registrations only after this module has been imported.
            from lightx2v.utils.registry_factory import SPARSE_OPERATOR_REGISTER

            fallback_cls = SPARSE_OPERATOR_REGISTER.get(self.fallback_operator)
            if fallback_cls is None:
                available = ", ".join(sorted(SPARSE_OPERATOR_REGISTER.keys()))
                raise KeyError(f"Unknown sparse fallback operator {self.fallback_operator!r}; available operators: {available}")
            self._fallback = fallback_cls(self.fallback_setting)
            if self._fallback.q_block_size != self.q_block_size or self._fallback.k_block_size != self.k_block_size:
                raise ValueError(f"musa_sparse fallback must use 64x128 blocks, got {self._fallback.q_block_size}x{self._fallback.k_block_size} from {self.fallback_operator!r}")
        return self._fallback

    def _unsupported(self, reason, q, k, v, mask, **kwargs):
        fallback = self._get_fallback()
        if fallback is None:
            raise NotImplementedError(
                f"musa_sparse cannot run this attention contract: {reason}. Set operator_setting.fallback_operator to a compatible 64x128 sparse operator if fallback is desired."
            )
        if reason not in self._warned_fallback_reasons:
            logger.warning(f"musa_sparse falling back to {self.fallback_operator}: {reason}")
            self._warned_fallback_reasons.add(reason)
        return fallback(q, k, v, mask, **kwargs)

    def _validate_qkv(self, q, k, v):
        if sparse_msa is None or sparse_msa_plan is None:
            return "mate.msa_interface.sparse_msa is not installed"
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            return f"Q/K/V must use flattened [tokens, heads, head_dim] layout, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        if k.shape != v.shape:
            return f"K and V shapes must match, got {tuple(k.shape)} and {tuple(v.shape)}"
        if q.shape[-1] != 128 or k.shape[-1] != 128:
            return f"head_dim must be 128, got Q={q.shape[-1]} and K={k.shape[-1]}"
        if q.dtype not in (torch.float16, torch.bfloat16):
            return f"dtype must be float16 or bfloat16, got {q.dtype}"
        if q.dtype != k.dtype or q.dtype != v.dtype:
            return f"Q/K/V dtypes must match, got {q.dtype}, {k.dtype}, {v.dtype}"
        if q.device != k.device or q.device != v.device:
            return f"Q/K/V devices must match, got {q.device}, {k.device}, {v.device}"

        num_q_heads = int(q.shape[1])
        num_kv_heads = int(k.shape[1])
        if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
            return f"local Hq ({num_q_heads}) must be divisible by Hkv ({num_kv_heads})"
        head_ratio = num_q_heads // num_kv_heads
        if head_ratio not in self._SUPPORTED_HEAD_RATIOS:
            return f"the installed MATE SQMMA kernel requires local Hq/Hkv ratio 1, 8, or 16, got {num_q_heads}/{num_kv_heads}={head_ratio}"
        return None

    def _validate_mask(self, mask, q_len, kv_len, num_q_heads, num_kv_heads):
        q_blocks = math.ceil(q_len / self.q_block_size)
        kv_blocks = math.ceil(kv_len / self.k_block_size)
        if mask.ndim != 4 or mask.shape[0] != 1:
            return f"mask must have shape [1, Hq or Hkv, Q_blocks, K_blocks], got {tuple(mask.shape)}"
        if int(mask.shape[1]) not in (num_q_heads, num_kv_heads):
            return f"mask head count must be Hq ({num_q_heads}) or Hkv ({num_kv_heads}), got {mask.shape[1]}"
        if tuple(mask.shape[-2:]) != (q_blocks, kv_blocks):
            return f"mask block grid must be {(q_blocks, kv_blocks)}, got {tuple(mask.shape[-2:])}"
        if kv_blocks < self.topk:
            return f"K has only {kv_blocks} blocks, fewer than configured topk={self.topk}"

        return None

    def _validate_block_indices(
        self,
        block_indices,
        q_len,
        kv_len,
        num_q_heads,
        num_kv_heads,
    ):
        q_blocks = math.ceil(q_len / self.q_block_size)
        kv_blocks = math.ceil(kv_len / self.k_block_size)
        if block_indices is None or block_indices.ndim != 4 or block_indices.shape[0] != 1:
            shape = None if block_indices is None else tuple(block_indices.shape)
            return f"block indices must have shape [1, Hq or Hkv, Q_blocks, topk], got {shape}"
        if int(block_indices.shape[1]) not in (num_q_heads, num_kv_heads):
            return f"block index head count must be Hq ({num_q_heads}) or Hkv ({num_kv_heads}), got {block_indices.shape[1]}"
        expected_tail = (q_blocks, self.topk)
        if tuple(block_indices.shape[-2:]) != expected_tail:
            return f"block index grid must end in {expected_tail}, got {tuple(block_indices.shape[-2:])}"
        if kv_blocks < self.topk:
            return f"K has only {kv_blocks} blocks, fewer than configured topk={self.topk}"
        return None

    def __call__(
        self,
        q,
        k,
        v,
        mask,
        block_indices=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        fallback_kwargs = dict(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            **kwargs,
        )

        reason = self._validate_qkv(q, k, v)
        if reason is not None:
            return self._unsupported(reason, q, k, v, mask, **fallback_kwargs)

        q_len, kv_len = int(q.shape[0]), int(k.shape[0])
        num_q_heads, num_kv_heads = int(q.shape[1]), int(k.shape[1])
        if cu_seqlens_q is not None and int(cu_seqlens_q.numel()) != 2:
            reason = "only a single flattened sequence is currently supported"
            return self._unsupported(reason, q, k, v, mask, **fallback_kwargs)
        if cu_seqlens_kv is not None and int(cu_seqlens_kv.numel()) != 2:
            reason = "only a single flattened KV sequence is currently supported"
            return self._unsupported(reason, q, k, v, mask, **fallback_kwargs)

        if mask is None:
            reason = self._validate_block_indices(block_indices, q_len, kv_len, num_q_heads, num_kv_heads)
        else:
            reason = self._validate_mask(mask, q_len, kv_len, num_q_heads, num_kv_heads)
        if reason is not None:
            return self._unsupported(reason, q, k, v, mask, **fallback_kwargs)

        causal = bool(kwargs.get("causal", False))
        softmax_scale = kwargs.get("softmax_scale")
        if self.compute_dtype == "mixed_bf16_fp8pv" and (q.dtype != torch.bfloat16 or causal or num_q_heads != num_kv_heads or self.topk not in self._native_mixed_topks):
            reason = f"mixed BF16/FP8 PV requires BF16 non-causal ratio-1 and native topk in {self._native_mixed_topks}"
            return self._unsupported(reason, q, k, v, mask, **fallback_kwargs)
        if self.compute_dtype == "int8_qk_fp8pv" and (q.dtype != torch.bfloat16 or causal or num_q_heads != num_kv_heads or self.topk not in self._native_mixed_topks):
            reason = f"INT8-QK/FP8-PV requires BF16 non-causal ratio-1 and native topk in {self._native_mixed_topks}"
            return self._unsupported(reason, q, k, v, mask, **fallback_kwargs)
        if self.compute_dtype == "int8_qk_block_fp8pv" and (q.dtype != torch.bfloat16 or causal or num_q_heads != num_kv_heads or self.topk not in self._native_mixed_topks):
            reason = f"block-INT8-QK/FP8-PV requires BF16 non-causal ratio-1 and native topk in {self._native_mixed_topks}"
            return self._unsupported(reason, q, k, v, mask, **fallback_kwargs)

        mask_heads = int(block_indices.shape[1]) if mask is None else int(mask.shape[1])
        if block_indices is None:
            block_indices = torch.topk(
                mask[0].to(dtype=torch.int32),
                self.topk,
                dim=-1,
                sorted=False,
            ).indices.unsqueeze(0)
        expected_shape = (1, mask_heads, math.ceil(q_len / self.q_block_size), self.topk)
        if tuple(block_indices.shape) != expected_shape:
            reason = f"block indices must have shape {expected_shape}, got {tuple(block_indices.shape)}"
            return self._unsupported(reason, q, k, v, mask, **fallback_kwargs)
        block_indexes = block_indices[0].permute(1, 0, 2)
        head_ratio = num_q_heads // num_kv_heads
        if causal or head_ratio != 1:
            block_indexes = block_indexes.repeat_interleave(
                self.q_block_size,
                dim=0,
            )[:q_len]
        block_indexes = block_indexes.contiguous().to(dtype=torch.int32)
        assert int(block_indexes.shape[1]) == mask_heads

        pad_kv = (-kv_len) % self.k_block_size
        if pad_kv:
            k = F.pad(k, (0, 0, 0, 0, 0, pad_kv))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_kv))
        num_pages = int(k.shape[0]) // self.k_block_size
        k = k.reshape(num_pages, self.k_block_size, num_kv_heads, 128).contiguous()
        v = v.reshape(num_pages, self.k_block_size, num_kv_heads, 128).contiguous()
        out = musa_sparse_msa(
            q.contiguous(),
            k,
            v,
            block_indexes,
            q_len,
            kv_len,
            num_q_heads,
            num_kv_heads,
            self.topk,
            self.q_block_size,
            causal,
            softmax_scale,
            self.validate_lut,
            {
                "bf16": 0,
                "fp8_e4m3": 1,
                "mixed_bf16_fp8pv": 2,
                "int8_qk_fp8pv": 3,
                "int8_qk_block_fp8pv": 4,
            }[self.compute_dtype],
            self.q_scale,
            self.k_scale,
            self.v_scale,
        )
        return out.reshape(q_len, -1)
