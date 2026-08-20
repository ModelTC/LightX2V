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
_VALID_DENSE_BACKENDS = ("flash_attn3", "sage_attn2", "torch_sdpa")
_VALID_COMPILE_MODES = ("default", "sm120_compile_once")
_FALLBACK_WARNINGS = set()
_KERNEL_LOGS = set()
_DENSE_GUARD_LOGS = set()
_DENSE_BACKEND_WARNINGS = set()


class _CompiledSolAttnWithKeywordStream:
    """Adapt CuTe's positional TVM-FFI stream argument to upstream calls."""

    def __init__(self, compiled):
        self.compiled = compiled

    def __call__(self, *args, **kwargs):
        stream = kwargs.pop("stream", None)
        if kwargs:
            return self.compiled(*args, **kwargs)
        if stream is not None:
            args = (*args, stream)
        return self.compiled(*args)

    def __getattr__(self, name):
        return getattr(self.compiled, name)


def _torch_stream_handle(stream):
    """Return a CUDA stream handle from legacy or unified PyTorch streams."""

    handle = getattr(stream, "cuda_stream", None)
    if handle is None:
        handle = stream.native_handle
        if callable(handle):
            handle = handle()
    return handle


def _install_sol_attn_runtime_compat(interface):
    """Install LightX2V compatibility for the pinned upstream SM120 backend."""

    if getattr(interface, "_lightx2v_runtime_compat", False):
        return

    def current_stream(device):
        import cuda.bindings.driver as cuda

        return cuda.CUstream(_torch_stream_handle(torch.cuda.current_stream(device)))

    interface._stream = current_stream
    original_compile_sm120 = getattr(interface, "_compile_sm120", None)
    if original_compile_sm120 is not None:

        def compile_sm120(key, *args, **kwargs):
            compiled, call_args = original_compile_sm120(key, *args, **kwargs)
            compiled = _CompiledSolAttnWithKeywordStream(compiled)
            interface._compiled[key] = compiled
            return compiled, call_args

        interface._compile_sm120 = compile_sm120
    interface._lightx2v_runtime_compat = True


def _parse_dense_layers(value):
    """Parse layer indices from JSON lists or Sol-Engine-style ranges."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return frozenset()
    if isinstance(value, (int, str)):
        value = [value]

    layers = set()
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError("sol_attn_setting.dense_layers must be a list of indices or a range string.") from exc

    for item in items:
        if isinstance(item, str):
            for part in item.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    start_text, end_text = part.split("-", 1)
                    start, end = int(start_text), int(end_text)
                    if start < 0 or end < start:
                        raise ValueError(f"Invalid dense layer range: {part!r}.")
                    layers.update(range(start, end + 1))
                else:
                    layer = int(part)
                    if layer < 0:
                        raise ValueError("Dense layer indices must be non-negative.")
                    layers.add(layer)
        else:
            layer = int(item)
            if layer < 0:
                raise ValueError("Dense layer indices must be non-negative.")
            layers.add(layer)
    return frozenset(layers)


@functools.lru_cache(maxsize=1)
def _load_sol_attn():
    """Import the optional backend only when it is selected by a config."""

    try:
        module = importlib.import_module("sol_attn")
    except ImportError as exc:
        raise ImportError("Sol-Attn is not installed. Run scripts/install_sol_attn.sh, then restart the LightX2V process.") from exc
    interface = importlib.import_module("sol_attn.interface")
    _install_sol_attn_runtime_compat(interface)
    return module.sol_attn


@torch.compiler.disable
def _run_sol_attn(q, k, v, *, scale, tau, thresh_type, kv_splits, sink_tokens, sink_start):
    """Keep CuTe DSL and TVM FFI calls outside TorchDynamo graphs."""

    return _load_sol_attn()(
        q,
        k,
        v,
        scale=scale,
        tau=tau,
        thresh_type=thresh_type,
        kv_splits=kv_splits,
        sink_tokens=sink_tokens,
        sink_start=sink_start,
    )


@torch.compiler.disable
def _run_sol_attn_sm120_compile_once(q, k, v, *, scale, tau, thresh_type, kv_splits, sink_tokens, sink_start):
    """Run the shape-polymorphic SM120 kernel with fixed Triton launch metadata."""

    # Importing the public backend first installs the PyTorch/CuTe stream
    # compatibility shim used by the pinned SM120 package.
    _load_sol_attn()
    from .sol_attn_sm120_compile_once import sol_attn_sm120_compile_once

    return sol_attn_sm120_compile_once(
        q,
        k,
        v,
        scale=scale,
        tau=tau,
        thresh_type=thresh_type,
        kv_splits=kv_splits,
        sink_tokens=sink_tokens,
        sink_start=sink_start,
    )


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
        self.dense_backend = None
        self.dense_backend_name = None
        self.set_config({})

    @staticmethod
    def _create_dense_backend(name):
        if name == "flash_attn3":
            from .flash_attn import FlashAttn3Weight

            return FlashAttn3Weight()
        if name == "sage_attn2":
            from .sage_attn import SageAttn2Weight

            return SageAttn2Weight()
        if name == "torch_sdpa":
            from .torch_sdpa import TorchSDPAWeight

            return TorchSDPAWeight()
        raise ValueError(f"sol_attn_setting.dense_backend must be one of {_VALID_DENSE_BACKENDS}, got {name!r}.")

    def set_config(self, config=None):
        self.config = dict(config or {})
        self.tau = float(self.config.get("tau", 1.0))
        self.thresh_type = str(self.config.get("thresh_type", "diag"))
        self.kv_splits = self.config.get("kv_splits", "auto")
        self.sink_tokens = int(self.config.get("sink_tokens", 0))
        self.sink_start = self.config.get("sink_start")
        self.reorder = str(self.config.get("reorder", "none")).lower()
        self.compile_mode = str(self.config.get("compile_mode", "default")).lower()
        self.strict = bool(self.config.get("strict", False))
        self.dense_steps = int(self.config.get("dense_steps", 0))
        self.dense_layers = _parse_dense_layers(self.config.get("dense_layers", ()))
        dense_backend_name = str(self.config.get("dense_backend", "flash_attn3")).lower()
        if dense_backend_name not in _VALID_DENSE_BACKENDS:
            raise ValueError(f"sol_attn_setting.dense_backend must be one of {_VALID_DENSE_BACKENDS}, got {dense_backend_name!r}.")
        if dense_backend_name != self.dense_backend_name:
            self.dense_backend = self._create_dense_backend(dense_backend_name)
            self.dense_backend_name = dense_backend_name

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
        if self.compile_mode not in _VALID_COMPILE_MODES:
            raise ValueError(f"sol_attn_setting.compile_mode must be one of {_VALID_COMPILE_MODES}, got {self.compile_mode!r}.")
        if self.compile_mode == "sm120_compile_once" and self.thresh_type != "diag":
            raise ValueError("sol_attn_setting.compile_mode='sm120_compile_once' requires thresh_type='diag'.")
        if self.compile_mode == "sm120_compile_once" and self.kv_splits not in ("auto", 1):
            raise ValueError("sol_attn_setting.compile_mode='sm120_compile_once' requires kv_splits='auto' or 1.")
        if self.dense_steps < 0:
            raise ValueError("sol_attn_setting.dense_steps must be non-negative.")

    def _strict_enabled(self):
        return self.strict or os.environ.get("SOL_ATTN_STRICT", "0") == "1"

    def _dense_guard(self, kwargs):
        """Keep quality-sensitive denoising steps and layers on dense attention."""

        if self.dense_steps:
            scheduler = kwargs.get("scheduler")
            step_index = getattr(scheduler, "step_index", None)
            if step_index is None:
                return True, "scheduler.step_index is unavailable"
            if int(step_index) < self.dense_steps:
                return True, "warmup_step"

        if self.dense_layers:
            block_idx = kwargs.get("block_idx")
            if block_idx is None:
                return True, "block_idx is unavailable"
            if int(block_idx) in self.dense_layers:
                return True, "dense_layer"

        return False, None

    def _log_dense_guard(self, reason):
        key = (reason, self.dense_steps, self.dense_layers, self.dense_backend_name)
        if key in _DENSE_GUARD_LOGS:
            return
        if reason in ("warmup_step", "dense_layer"):
            logger.info(
                "Sol-Attn dense guard active: dense_steps={}, dense_layers={}, backend={}.",
                self.dense_steps,
                sorted(self.dense_layers),
                self.dense_backend_name,
            )
        else:
            logger.warning(
                "Sol-Attn dense guard metadata missing ({}); using dense attention for safety.",
                reason,
            )
        _DENSE_GUARD_LOGS.add(key)

    def _dense_guard_attention(
        self,
        q,
        k,
        v,
        *,
        drop_rate,
        attn_mask,
        causal,
        scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
    ):
        """Run quality-guarded calls through the configured dense backend."""

        dense_kwargs = {
            "drop_rate": drop_rate,
            "attn_mask": attn_mask,
            "causal": causal,
            "scale": scale,
        }
        if float(drop_rate) != 0.0 or attn_mask is not None:
            return _dense_attention(q, k, v, **dense_kwargs)

        query_len = q.shape[0] if q.ndim == 3 else q.shape[1]
        key_len = k.shape[0] if k.ndim == 3 else k.shape[1]
        try:
            out = self.dense_backend.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q or query_len,
                max_seqlen_kv=max_seqlen_kv or key_len,
                causal=causal,
                softmax_scale=scale,
            )
            if q.ndim == 4:
                return out.reshape(q.shape[0], q.shape[1], -1)
            return out
        except Exception as exc:
            warning_key = (self.dense_backend_name, type(exc).__name__, str(exc))
            if warning_key not in _DENSE_BACKEND_WARNINGS:
                logger.warning(
                    "Sol-Attn dense guard could not use {} ({}: {}); falling back to torch SDPA.",
                    self.dense_backend_name,
                    type(exc).__name__,
                    exc,
                )
                _DENSE_BACKEND_WARNINGS.add(warning_key)
            return _dense_attention(q, k, v, **dense_kwargs)

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
        use_dense, guard_reason = self._dense_guard(kwargs)
        if use_dense:
            self._log_dense_guard(guard_reason)
            return self._dense_guard_attention(
                q,
                k,
                v,
                drop_rate=drop_rate,
                attn_mask=attn_mask,
                causal=causal,
                scale=scale,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
            )

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
        globally_reordered = bool(kwargs.get("sol_morton_preordered", False))
        if self.reorder == "morton3d" and not globally_reordered:
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
            run_kernel = _run_sol_attn_sm120_compile_once if self.compile_mode == "sm120_compile_once" else _run_sol_attn
            out = run_kernel(
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
        reorder_mode = "morton3d_global" if globally_reordered else self.reorder
        kernel_log_key = (arch, self.tau, self.thresh_type, self._resolve_kv_splits(q, self.kv_splits), reorder_mode, self.compile_mode)
        if kernel_log_key not in _KERNEL_LOGS:
            logger.info(
                "Sol-Attn active: SM{}{}, tau={}, thresh_type={}, kv_splits={}, reorder={}, compile_mode={}.",
                arch[0],
                arch[1],
                self.tau,
                self.thresh_type,
                self._resolve_kv_splits(q, self.kv_splits),
                reorder_mode,
                self.compile_mode,
            )
            _KERNEL_LOGS.add(kernel_log_key)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        return out.squeeze(0) if input_was_3d else out
