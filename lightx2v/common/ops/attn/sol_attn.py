import functools
import importlib
import math
import os

import torch
import torch.nn.functional as F
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate

HEAD_DIM = 128
_VALID_KV_SPLITS = (1, 2, 4)
_FALLBACK_WARNINGS = set()
_KERNEL_LOGS = set()


@functools.lru_cache(maxsize=1)
def _load_sol_attn():
    """Import the optional backend only when it is selected by a config."""

    try:
        module = importlib.import_module("sol_attn")
    except ImportError as exc:
        raise ImportError(
            "Sol-Attn is not installed. Run scripts/install_sol_attn.sh, then "
            "restart the LightX2V process."
        ) from exc
    return module.sol_attn


@functools.lru_cache(maxsize=1)
def _cute_runtime_available():
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return True


@functools.lru_cache(maxsize=32)
def _morton3d_indices_cpu(grid):
    """Build the same x/y/z-interleaved Morton order used by Sol-Engine."""

    frames, height, width = grid
    total = frames * height * width
    linear = torch.arange(total, dtype=torch.long)
    frame_area = height * width
    z = linear // frame_area
    rem = linear - z * frame_area
    y = rem // width
    x = rem - y * width

    def part1by2(value):
        value = value & 0x1FFFFF
        value = (value | (value << 32)) & 0x1F00000000FFFF
        value = (value | (value << 16)) & 0x1F0000FF0000FF
        value = (value | (value << 8)) & 0x100F00F00F00F00F
        value = (value | (value << 4)) & 0x10C30C30C30C30C3
        return (value | (value << 2)) & 0x1249249249249249

    code = part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)
    permutation = linear[torch.argsort(code)]
    return permutation, torch.argsort(permutation)


@functools.lru_cache(maxsize=64)
def _morton3d_indices_on_device(grid, device_string):
    permutation, inverse = _morton3d_indices_cpu(grid)
    device = torch.device(device_string)
    return permutation.to(device=device), inverse.to(device=device)


def _morton3d_indices(grid, device):
    grid = tuple(int(value) for value in grid)
    return _morton3d_indices_on_device(grid, str(device))


def _dense_attention(q, k, v, *, drop_rate=0.0, attn_mask=None, causal=False, scale=None):
    input_was_3d = q.ndim == 3
    if input_was_3d:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.to(q.dtype)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=float(drop_rate),
        is_causal=bool(causal),
        scale=scale,
    ).transpose(1, 2)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    return out.squeeze(0) if input_was_3d else out


@ATTN_WEIGHT_REGISTER("sol_attn")
class SolAttnWeight(AttnWeightTemplate):
    """LightX2V adapter for the public Sol-Attn BTHD forward API."""

    def __init__(self):
        self.config = {}
        self.set_config({})

    def set_config(self, config=None):
        self.config = dict(config or {})
        self.tau = float(self.config.get("tau", 1.0))
        self.thresh_type = str(self.config.get("thresh_type", "diag"))
        self.kv_splits = self.config.get("kv_splits", "auto")
        self.sink_tokens = int(self.config.get("sink_tokens", 0))
        self.sink_start = self.config.get("sink_start")
        self.reorder = str(self.config.get("reorder", "none")).lower()
        self.strict = bool(self.config.get("strict", False))

        if not math.isfinite(self.tau) or self.tau < 0:
            raise ValueError("sol_attn_setting.tau must be a finite non-negative number.")
        if self.thresh_type not in ("diag", "exact"):
            raise ValueError("sol_attn_setting.thresh_type must be 'diag' or 'exact'.")
        if self.kv_splits != "auto":
            self.kv_splits = int(self.kv_splits)
            if self.kv_splits not in _VALID_KV_SPLITS:
                raise ValueError("sol_attn_setting.kv_splits must be 'auto', 1, 2, or 4.")
        if self.sink_tokens < 0:
            raise ValueError("sol_attn_setting.sink_tokens must be non-negative.")
        if self.sink_start is not None:
            self.sink_start = int(self.sink_start)
            if self.sink_start < 0:
                raise ValueError("sol_attn_setting.sink_start must be non-negative or null.")
        if self.reorder not in ("none", "morton3d"):
            raise ValueError("sol_attn_setting.reorder must be 'none' or 'morton3d'.")

    def _strict_enabled(self):
        return self.strict or os.environ.get("SOL_ATTN_STRICT", "0") == "1"

    @staticmethod
    def _ineligibility_reason(q, k, v, *, drop_rate, attn_mask, causal, cu_seqlens_q, cu_seqlens_kv):
        if q.ndim not in (3, 4):
            return f"expected [T,H,D] or [B,T,H,D], got q.ndim={q.ndim}"
        if q.shape != k.shape or q.shape != v.shape:
            return "q, k, and v must have the same shape (Sol-Attn is self-attention only)"
        if q.shape[-1] != HEAD_DIM:
            return f"head dimension must be {HEAD_DIM}, got {q.shape[-1]}"
        if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v)):
            return "q, k, and v must use torch.bfloat16"
        if not q.is_cuda or k.device != q.device or v.device != q.device:
            return "q, k, and v must be on the same CUDA device"
        if float(drop_rate) != 0.0:
            return "dropout is unsupported"
        if attn_mask is not None:
            return "attention masks are unsupported"
        if causal:
            return "causal attention is unsupported"
        if q.ndim == 3:
            for name, cu_seqlens in (("q", cu_seqlens_q), ("kv", cu_seqlens_kv)):
                if cu_seqlens is not None and cu_seqlens.numel() > 2:
                    return f"packed multi-sequence cu_seqlens_{name} is unsupported"
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v)):
            return "the released Sol-Attn kernels are forward-only"
        try:
            major, minor = torch.cuda.get_device_capability(q.device)
        except Exception as exc:
            return f"could not query CUDA compute capability: {exc}"
        if major < 8:
            return f"compute capability >= 8.0 is required, got SM{major}{minor}"
        return None

    @staticmethod
    def _resolve_kv_splits(q, value):
        if value != "auto":
            return int(value)
        arch = tuple(torch.cuda.get_device_capability(q.device))
        if arch == (9, 0) and q.shape[1] >= 65536 and _cute_runtime_available():
            return 4
        return 1

    def _fallback_or_raise(self, reason, q, k, v, dense_kwargs, exc=None):
        message = f"Sol-Attn unavailable for this call: {reason}"
        if self._strict_enabled():
            if exc is not None:
                raise RuntimeError(message) from exc
            raise RuntimeError(message)
        if reason not in _FALLBACK_WARNINGS:
            logger.warning("{}; falling back to torch SDPA.", message)
            _FALLBACK_WARNINGS.add(reason)
        return _dense_attention(q, k, v, **dense_kwargs)

    def apply(
        self,
        q,
        k,
        v,
        drop_rate=0,
        attn_mask=None,
        causal=False,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        scale = kwargs.get("softmax_scale", kwargs.get("scale"))
        dense_kwargs = {
            "drop_rate": drop_rate,
            "attn_mask": attn_mask,
            "causal": causal,
            "scale": scale,
        }
        reason = self._ineligibility_reason(
            q,
            k,
            v,
            drop_rate=drop_rate,
            attn_mask=attn_mask,
            causal=causal,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
        )
        if reason is not None:
            return self._fallback_or_raise(reason, q, k, v, dense_kwargs)

        input_was_3d = q.ndim == 3
        if input_was_3d:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        inverse = None
        if self.reorder == "morton3d":
            grid = kwargs.get("grid_sizes")
            if grid is None or math.prod(int(value) for value in grid) != q.shape[1]:
                reason = f"morton3d reorder requires grid_sizes whose product equals T={q.shape[1]}, got {grid}"
                original = (q.squeeze(0), k.squeeze(0), v.squeeze(0)) if input_was_3d else (q, k, v)
                return self._fallback_or_raise(reason, *original, dense_kwargs)
            permutation, inverse = _morton3d_indices(grid, q.device)
            q = q.index_select(1, permutation)
            k = k.index_select(1, permutation)
            v = v.index_select(1, permutation)

        try:
            kernel = _load_sol_attn()
            out = kernel(
                q,
                k,
                v,
                scale=scale,
                tau=self.tau,
                thresh_type=self.thresh_type,
                kv_splits=self._resolve_kv_splits(q, self.kv_splits),
                sink_tokens=self.sink_tokens,
                sink_start=self.sink_start,
            )
        except Exception as exc:
            original = (q, k, v)
            if inverse is not None:
                original = tuple(tensor.index_select(1, inverse) for tensor in original)
            if input_was_3d:
                original = tuple(tensor.squeeze(0) for tensor in original)
            reason = f"{type(exc).__name__}: {exc}"
            return self._fallback_or_raise(reason, *original, dense_kwargs, exc=exc)

        if inverse is not None:
            out = out.index_select(1, inverse)
        arch = torch.cuda.get_device_capability(q.device)
        kernel_log_key = (arch, self.tau, self.thresh_type, self._resolve_kv_splits(q, self.kv_splits), self.reorder)
        if kernel_log_key not in _KERNEL_LOGS:
            logger.info(
                "Sol-Attn active: SM{}{}, tau={}, thresh_type={}, kv_splits={}, reorder={}.",
                arch[0],
                arch[1],
                self.tau,
                self.thresh_type,
                self._resolve_kv_splits(q, self.kv_splits),
                self.reorder,
            )
            _KERNEL_LOGS.add(kernel_log_key)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        return out.squeeze(0) if input_was_3d else out
