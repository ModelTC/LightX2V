"""Compile-once Sol-Attn path for shape-polymorphic SM120 kernels.

The released SM120 CuTe kernel accepts dynamic tensor layouts, but the public
Sol-Attn interface caches it by the exact token count.  Its Triton
preprocessing kernels also autotune by that count.  This module keeps the
logical token count dynamic in both places: the CuTe callable is compiled once
per device/layout, and the preprocessing kernels use one fixed RTX 5090
configuration without autotuning.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.metadata
import os
import platform
import sys
import threading
from pathlib import Path

import torch
import triton
import triton.language as tl
from loguru import logger

BLOCK_SIZE = 64
HEAD_DIM = 128
THRESHOLD_GROUP_SIZE = 64
_COMPILE_LOCK = threading.Lock()
_PREPARE_INIT_LOCK = threading.Lock()
_PREPARE_READY = set()
_COMPILE_KEY_PREFIX = "lightx2v_sm120_compile_once"
_PERSISTENT_CACHE_SCHEMA = "sm120-dynamic-t-v1"
_AOT_FUNCTION_NAME = "lightx2v_sol_attn_sm120"
_LOADED_AOT_MODULES = []
_PERSISTENT_CACHE_LOGS = set()


@triton.jit(
    do_not_specialize=("T", "N"),
    do_not_specialize_on_alignment=("T", "N"),
)
def _reduce_kv_kernel(
    k,
    v,
    kc,
    vc,
    T,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    tokens = block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    valid = tokens < T
    offsets = ((batch * T + tokens[:, None]).to(tl.int64) * H + head) * D + dims[None, :]
    k_values = tl.load(k + offsets, mask=valid[:, None], other=0.0)
    v_values = tl.load(v + offsets, mask=valid[:, None], other=0.0)
    block_len = tl.minimum(BLOCK, T - block * BLOCK).to(tl.float32)
    summary_offsets = ((batch * N + block) * H + head) * D + dims
    tl.store(kc + summary_offsets, tl.sum(k_values, axis=0) / block_len)
    tl.store(vc + summary_offsets, tl.sum(v_values, axis=0))


@triton.jit(
    do_not_specialize=("N",),
    do_not_specialize_on_alignment=("N",),
)
def _reduce_kc_stats_kernel(
    kc,
    kc_mean,
    kc_var_diag,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
):
    batch_head = tl.program_id(0)
    batch, head = batch_head // H, batch_head % H
    blocks = tl.max_contiguous(tl.arange(0, GROUP), GROUP)
    dims = tl.arange(0, D)
    total = tl.zeros((D,), dtype=tl.float32)
    total_sq = tl.zeros((D,), dtype=tl.float32)
    count = tl.full((), 0.0, dtype=tl.float32)
    for start in tl.range(0, N, GROUP):
        block_indices = start + blocks
        valid = block_indices < N
        offsets = ((batch * N + block_indices[:, None]) * H + head) * D + dims[None, :]
        values = tl.load(
            kc + offsets,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        total += tl.sum(values, axis=0)
        total_sq += tl.sum(values * values, axis=0)
        count += tl.sum(valid.to(tl.float32), axis=0)
    mean = total / count
    variance = tl.maximum(total_sq / count - mean * mean, 0.0)
    tl.store(kc_mean + batch_head * D + dims, mean)
    tl.store(kc_var_diag + batch_head * D + dims, variance)


@triton.jit(
    do_not_specialize=("T", "N"),
    do_not_specialize_on_alignment=("T", "N"),
)
def _diag_threshold_kernel(
    q,
    kc_mean,
    kc_var_diag,
    threshold,
    scale,
    tau,
    T,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    tokens = q_block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    valid = tokens < T
    offsets = ((batch * T + tokens[:, None]).to(tl.int64) * H + head) * D + dims[None, :]
    q_values = tl.load(q + offsets, mask=valid[:, None], other=0.0)
    q_len = tl.minimum(BLOCK, T - q_block * BLOCK).to(tl.float32)
    q_centroid = tl.sum(q_values.to(tl.float32), axis=0) / q_len
    mean_kc = tl.load(kc_mean + batch_head * D + dims)
    var_kc = tl.load(kc_var_diag + batch_head * D + dims)
    log2_scale = scale * 1.4426950408889634
    mean = tl.sum(q_centroid * mean_kc, axis=0) * log2_scale
    variance = tl.sum(q_centroid * q_centroid * var_kc, axis=0)
    variance *= log2_scale * log2_scale
    std = tl.sqrt(tl.maximum(variance, 0.0) + 1.0e-6)
    tl.store(
        threshold + (batch * N + q_block) * H + head,
        mean + tau * std,
    )


def _prepare_diag_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    kc = torch.empty(
        (batch, blocks, heads, head_dim),
        device=q.device,
        dtype=torch.bfloat16,
    )
    vc = torch.empty_like(kc)
    _reduce_kv_kernel[(blocks, batch * heads)](
        k,
        v,
        kc,
        vc,
        tokens,
        blocks,
        H=heads,
        D=head_dim,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )

    batch_heads = batch * heads
    kc_mean = torch.empty(
        (batch_heads, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    kc_var_diag = torch.empty_like(kc_mean)
    _reduce_kc_stats_kernel[(batch_heads,)](
        kc,
        kc_mean,
        kc_var_diag,
        blocks,
        H=heads,
        D=head_dim,
        GROUP=THRESHOLD_GROUP_SIZE,
        num_warps=4,
        num_stages=1,
    )

    threshold = torch.empty(
        (batch, blocks, heads),
        device=q.device,
        dtype=torch.float32,
    )
    _diag_threshold_kernel[(blocks, batch_heads)](
        q,
        kc_mean,
        kc_var_diag,
        threshold,
        scale,
        tau,
        tokens,
        blocks,
        H=heads,
        D=head_dim,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    return kc, vc, threshold


def prepare_diag(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Sol block summaries with one fixed, dynamic-length Triton binary."""

    prepare_key = (
        q.device.index,
        q.shape[2],
        q.shape[3],
        q.dtype,
    )
    if prepare_key not in _PREPARE_READY:
        # Triton's disk/in-memory cache is thread-safe, but it can still make
        # concurrent cache misses wait through duplicate launcher setup.  Only
        # the first call for a layout needs serialization.
        with _PREPARE_INIT_LOCK:
            if prepare_key not in _PREPARE_READY:
                result = _prepare_diag_impl(
                    q,
                    k,
                    v,
                    tau=tau,
                    scale=scale,
                )
                _PREPARE_READY.add(prepare_key)
                return result
    return _prepare_diag_impl(q, k, v, tau=tau, scale=scale)


def _compile_key(q: torch.Tensor, arch, kv_splits: int):
    """Return the SM120 layout key, deliberately excluding token count."""

    batch, _, heads, head_dim = q.shape
    return (
        _COMPILE_KEY_PREFIX,
        q.device.index,
        arch,
        batch,
        heads,
        head_dim,
        q.dtype,
        kv_splits,
    )


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _sm120_source_digest(interface) -> str:
    """Fingerprint the external kernel sources that produced the AOT object."""

    package_root = Path(interface.__file__).resolve().parent
    source_files = [package_root / "common" / "runtime.py"]
    source_files.extend(sorted((package_root / "sm120").rglob("*.py")))
    digest = hashlib.sha256()
    for source_file in source_files:
        digest.update(str(source_file.relative_to(package_root)).encode())
        digest.update(source_file.read_bytes())
    return digest.hexdigest()


def _persistent_cache_fingerprint(q: torch.Tensor, arch, kv_splits: int, interface) -> str:
    """Identify ABI/kernel changes while deliberately excluding T and device id."""

    batch, _, heads, head_dim = q.shape
    fields = (
        _PERSISTENT_CACHE_SCHEMA,
        platform.machine(),
        f"python-{sys.version_info.major}.{sys.version_info.minor}",
        f"torch-{torch.__version__}",
        f"cuda-{torch.version.cuda}",
        f"cutlass-{_distribution_version('nvidia-cutlass-dsl')}",
        f"tvm-ffi-{_distribution_version('apache-tvm-ffi')}",
        f"arch-{arch[0]}{arch[1]}",
        f"batch-{batch}",
        f"heads-{heads}",
        f"head-dim-{head_dim}",
        f"dtype-{q.dtype}",
        f"kv-splits-{kv_splits}",
        _sm120_source_digest(interface),
    )
    return hashlib.sha256("\n".join(map(str, fields)).encode()).hexdigest()[:24]


def _persistent_cache_root() -> Path:
    override = os.environ.get("LIGHTX2V_SOL_ATTN_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "lightx2v" / "sol_attn"
    return Path.home() / ".cache" / "lightx2v" / "sol_attn"


def _persistent_artifact_path(q: torch.Tensor, arch, kv_splits: int, interface) -> Path:
    fingerprint = _persistent_cache_fingerprint(q, arch, kv_splits, interface)
    return _persistent_cache_root() / f"{_AOT_FUNCTION_NAME}_{fingerprint}.o"


@contextlib.contextmanager
def _persistent_artifact_lock(artifact_path: Path):
    """Serialize the first compile across all torchrun ranks on this host."""

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_path.with_suffix(artifact_path.suffix + ".lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_persistent_compiled(artifact_path: Path):
    # Importing this module installs CuTe's AOT load provider as a side effect.
    import cutlass.cute.export  # noqa: F401
    from cutlass.cute.runtime import load_module

    from .sol_attn import _CompiledSolAttnWithKeywordStream

    module = load_module(str(artifact_path), enable_tvm_ffi=True)
    compiled = _CompiledSolAttnWithKeywordStream(getattr(module, _AOT_FUNCTION_NAME))
    # The function retains its execution engine, and this explicit reference also
    # prevents a future TVM-FFI implementation from unloading the object early.
    _LOADED_AOT_MODULES.append(module)
    return compiled


def _export_persistent_compiled(compiled, artifact_path: Path) -> None:
    raw_compiled = getattr(compiled, "compiled", compiled)
    if not hasattr(raw_compiled, "export_to_c"):
        raise TypeError(f"CuTe compiled function {type(raw_compiled)!r} cannot be exported")

    temporary_path = artifact_path.with_name(f".{artifact_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.o")
    try:
        raw_compiled.export_to_c(
            str(temporary_path),
            function_name=_AOT_FUNCTION_NAME,
            export_only_tvm_ffi_symbols=True,
        )
        with temporary_path.open("rb") as artifact_file:
            os.fsync(artifact_file.fileno())
        os.replace(temporary_path, artifact_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _log_persistent_cache(status: str, artifact_path: Path) -> None:
    key = (status, artifact_path)
    if key not in _PERSISTENT_CACHE_LOGS:
        logger.info("Sol-Attn SM120 persistent cache {}: {}", status, artifact_path)
        _PERSISTENT_CACHE_LOGS.add(key)


def _load_or_compile_persistent(
    interface,
    key,
    tensors,
    scale,
    sink_start_block,
    sink_end_block,
    stream,
    artifact_path: Path,
):
    """Load an AOT object, or let exactly one rank compile and publish it."""

    with _persistent_artifact_lock(artifact_path):
        if artifact_path.is_file():
            try:
                compiled = _load_persistent_compiled(artifact_path)
                interface._compiled[key] = compiled
                _log_persistent_cache("hit", artifact_path)
                return compiled, interface._to_cute_tensors(tensors)
            except Exception as exc:
                logger.warning(
                    "Sol-Attn persistent cache load failed ({}: {}); rebuilding {}.",
                    type(exc).__name__,
                    exc,
                    artifact_path,
                )
                artifact_path.unlink(missing_ok=True)

        _log_persistent_cache("miss; compiling once", artifact_path)
        compiled, args = interface._compile_sm120(
            key,
            tensors,
            scale,
            sink_start_block,
            sink_end_block,
            stream,
        )
        try:
            _export_persistent_compiled(compiled, artifact_path)
        except Exception as exc:
            # The freshly compiled in-memory callable remains valid.  AOT export
            # failure should not turn an otherwise supported attention call into
            # an inference failure.
            logger.warning(
                "Sol-Attn persistent cache export failed ({}: {}); continuing with the in-memory kernel.",
                type(exc).__name__,
                exc,
            )
        return compiled, args


def sol_attn_sm120_compile_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    thresh_type: str = "diag",
    kv_splits: int = 1,
    sink_tokens: int = 0,
    sink_start: int | None = None,
) -> torch.Tensor:
    """Run the RTX 5090 Sol-Attn binary without specializing on input length."""

    # Also make the direct function API safe: this installs the pinned
    # package's PyTorch stream/TVM-FFI compatibility shim before private SM120
    # helpers are used.  The normal SolAttnWeight path has already done this.
    from .sol_attn import _load_sol_attn

    _load_sol_attn()
    import sol_attn.interface as interface

    arch = interface._validate_inputs(
        q,
        k,
        v,
        thresh_type,
        sink_tokens,
        sink_start,
    )
    if arch != (12, 0):
        raise RuntimeError(f"sm120_compile_once requires an RTX 50-series SM120 GPU; got SM{arch[0]}{arch[1]}")
    if thresh_type != "diag":
        raise ValueError("sm120_compile_once currently requires thresh_type='diag'")
    if kv_splits != 1:
        raise ValueError("sm120_compile_once requires kv_splits=1")

    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    tau = float(tau)
    batch, tokens, heads, _ = q.shape

    with torch.cuda.device(q.device):
        kc, vc, threshold = prepare_diag(
            q,
            k,
            v,
            scale=scale,
            tau=tau,
        )
        output = torch.empty_like(v)
        lse = torch.empty(
            (batch, tokens, heads),
            device=q.device,
            dtype=torch.float32,
        )
        stream = interface._stream(q.device)
        sink_start_block, sink_end_block = interface._sink_block_range(
            tokens,
            sink_start,
            sink_tokens,
        )
        tensors = [q, k, v, output, kc, vc, threshold, lse]
        key = _compile_key(q, arch, kv_splits)
        artifact_path = _persistent_artifact_path(q, arch, kv_splits, interface)

        compiled = interface._compiled.get(key)
        if compiled is None:
            with _COMPILE_LOCK:
                compiled = interface._compiled.get(key)
                if compiled is None:
                    compiled, args = _load_or_compile_persistent(
                        interface,
                        key,
                        tensors,
                        scale,
                        sink_start_block,
                        sink_end_block,
                        stream,
                        artifact_path,
                    )
                else:
                    args = interface._to_cute_tensors(tensors)
        else:
            args = interface._to_cute_tensors(tensors)

        compiled(
            *args,
            scale,
            sink_start_block,
            sink_end_block,
            stream=stream,
        )
    return output


__all__ = ["prepare_diag", "sol_attn_sm120_compile_once"]
