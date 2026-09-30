"""SOL attention for S5000: KV preprocessing and native MP31 forward.

Adapted from NVlabs/Sana, commit 6c2f582ba9681dfe352ae4ca23b6f0030cf120cf.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import functools
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 64
HEAD_DIM = 128
THRESHOLD_GROUP_SIZE = 64
SUMMARY_PAD = 64
_ABI_VERSION = 2
_HEADER_SUFFIXES = {".h", ".hpp", ".cuh", ".inl", ".mu"}


def _validate_qkv(q, k, v):
    if any(not isinstance(x, torch.Tensor) for x in (q, k, v)):
        raise TypeError("MUSA SOL attention requires torch.Tensor q, k, and v")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("MUSA SOL attention requires matching q/k/v shapes [B, T, H, 128]")
    if any(size <= 0 for size in q.shape[:3]) or q.shape[3] != HEAD_DIM:
        raise ValueError("MUSA SOL attention requires B, T, H > 0 and head dimension 128")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("MUSA SOL attention requires BF16 q, k, and v")
    musa_device = q.device.type == "musa" or (os.environ.get("PLATFORM") == "musa" and q.device.type in ("cuda", "privateuseone"))
    if not musa_device or k.device != q.device or v.device != q.device:
        raise ValueError("MUSA SOL attention requires q, k, and v on the same MUSA device")
    if not all(x.is_contiguous() for x in (q, k, v)):
        raise ValueError("MUSA SOL attention requires contiguous BTHD q, k, and v")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise ValueError("MUSA SOL attention is forward-only")
    return q.shape[:3]


def _validate_inputs(q, k, v, *, thresh_type, kv_splits, sink_tokens, sink_start):
    _validate_qkv(q, k, v)
    if thresh_type not in ("diag", "exact"):
        raise ValueError("MUSA SOL attention thresh_type must be 'diag' or 'exact'")
    if type(kv_splits) is not int or kv_splits != 1:
        raise ValueError("MUSA SOL attention currently supports only kv_splits=1")
    tokens = q.shape[1]
    if type(sink_tokens) is not int or not 0 <= sink_tokens <= tokens:
        raise ValueError("MUSA SOL attention sink_tokens must be an integer in [0, T]")
    if sink_start is not None:
        if type(sink_start) is not int or not 0 <= sink_start <= tokens or sink_start + sink_tokens > tokens:
            raise ValueError("MUSA SOL attention sink range must lie within [0, T]")


def _hash_file(digest, path: Path, label: str) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\0")


def _build_spec() -> dict:
    source = Path(__file__).resolve().parent / "kernels" / "sol_kernel.mu"
    if not source.is_file():
        raise RuntimeError(f"MUSA native SOL source is missing: {source}")
    sdk = Path(os.environ.get("MUSA_HOME", "/usr/local/musa")).resolve()
    compiler_name = os.environ.get("LIGHTX2V_SOL_NATIVE_MCC", str(sdk / "bin" / "mcc"))
    compiler = shutil.which(compiler_name)
    if compiler is None:
        raise RuntimeError(f"MUSA native SOL requires mcc; cannot execute {compiler_name!r}. Set MUSA_HOME to the MUSA SDK or LIGHTX2V_SOL_NATIVE_MCC to its compiler.")
    mutlass = Path(os.environ.get("LIGHTX2V_MUTLASS_ROOT", os.environ.get("MUTLASS_ROOT", "/opt/mate/mate/data/mutlass"))).resolve()
    include_dirs = [sdk / "include", mutlass / "include", mutlass / "experimental" / "fmha"]
    for directory in include_dirs:
        if not directory.is_dir():
            raise RuntimeError(f"MUSA native SOL include directory is missing: {directory}. Set MUSA_HOME and LIGHTX2V_MUTLASS_ROOT to the installed SDK and MUTLASS source tree.")
    library_dirs = [directory for directory in (sdk / "lib", sdk / "lib64") if directory.is_dir()]
    if not library_dirs:
        raise RuntimeError(f"MUSA native SOL cannot find SDK libraries under {sdk}/lib or {sdk}/lib64")
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=False)
    if version.returncode:
        raise RuntimeError(f"MUSA native SOL compiler version check failed:\n{version.stdout}{version.stderr}")

    flags = [
        "-std=c++17",
        "-O3",
        "-shared",
        "-fPIC",
        "--cuda-gpu-arch=mp_31",
        "-fmusa-flush-denormals-to-zero",
        "-ffast-math",
    ]
    digest = hashlib.sha256()
    # Include transitive headers in the build cache key.
    for root, label in ((source.parent, "kernels"), (mutlass / "include", "mutlass/include"), (mutlass / "experimental" / "fmha", "mutlass/fmha")):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in _HEADER_SUFFIXES:
                _hash_file(digest, path, f"{label}/{path.relative_to(root)}")
    for name in ("musa.h", "musa_runtime.h", "musa_runtime_api.h", "musa_bf16.h", "musa_fp16.h", "mtgpu_bf16.h"):
        header = sdk / "include" / name
        if header.is_file():
            _hash_file(digest, header, f"sdk/{name}")
    spec = {
        "abi_version": _ABI_VERSION,
        "source": str(source),
        "source_digest": digest.hexdigest(),
        "compiler": str(Path(compiler).resolve()),
        "compiler_version": (version.stdout + version.stderr).strip(),
        "sdk": str(sdk),
        "mutlass": str(mutlass),
        "architecture": "mp_31",
        "flags": flags,
        "include_dirs": [str(path) for path in include_dirs],
        "library_dirs": [str(path) for path in library_dirs],
        "libraries": ["musart", "musa"],
    }
    spec["cache_key"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()
    return spec


def _acquire_build_lock(lock, *, timeout: float | None = None) -> None:
    """Wait for the cache build lock."""
    timeout = float(os.environ.get("LIGHTX2V_SOL_NATIVE_LOCK_TIMEOUT", "600")) if timeout is None else float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("LIGHTX2V_SOL_NATIVE_LOCK_TIMEOUT must be a finite positive number of seconds")
    deadline = time.monotonic() + timeout
    delay = 0.05
    while True:
        try:
            # Shared FUSE mounts may return EAGAIN for blocking flock too.
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES, errno.EINTR):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out after {timeout:g}s waiting for MUSA native SOL build lock {lock.name}") from error
            time.sleep(min(delay, remaining))
            delay = min(delay * 1.5, 0.5)


def _build_library(spec: dict) -> tuple[Path, dict]:
    default_cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "lightx2v" / "sol_native"
    cache_root = Path(os.environ.get("LIGHTX2V_SOL_NATIVE_CACHE", str(default_cache))).expanduser()
    build_dir = cache_root / spec["cache_key"]
    build_dir.mkdir(parents=True, exist_ok=True)
    library = build_dir / "sol_forward.so"
    metadata_path = build_dir / "build.json"
    log_path = build_dir / "build.log"
    # TP ranks share the build cache; publish only a complete library.
    with (build_dir / "build.lock").open("a+") as lock:
        _acquire_build_lock(lock)
        if library.is_file() and metadata_path.is_file():
            return library, json.loads(metadata_path.read_text())
        with tempfile.TemporaryDirectory(prefix="building-", dir=build_dir) as temporary:
            temporary_dir = Path(temporary)
            temporary_library = temporary_dir / library.name
            command = [spec["compiler"], *spec["flags"]]
            for include in spec["include_dirs"]:
                command.extend(["-I", include])
            command.append(spec["source"])
            for directory in spec["library_dirs"]:
                command.extend(["-L", directory, f"-Wl,-rpath,{directory}"])
            command.extend(f"-l{name}" for name in spec["libraries"])
            command.extend(["-o", str(temporary_library)])
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
            temporary_log = temporary_dir / log_path.name
            temporary_log.write_text(result.stdout)
            os.replace(temporary_log, log_path)
            if result.returncode or not temporary_library.is_file():
                raise RuntimeError(f"MUSA native SOL build failed (exit {result.returncode}); compiler log: {log_path}\n{result.stdout[-12000:]}")
            metadata = {**spec, "library_path": str(library), "build_log": str(log_path), "build_command": command}
            temporary_metadata = temporary_dir / metadata_path.name
            temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
            os.replace(temporary_metadata, metadata_path)
            os.replace(temporary_library, library)
            return library, metadata


@functools.lru_cache(maxsize=1)
def _load_library() -> tuple[ctypes.CDLL, dict]:
    library_path, metadata = _build_library(_build_spec())
    try:
        library = ctypes.CDLL(str(library_path))
    except OSError as error:
        raise RuntimeError(f"Unable to load MUSA native SOL library {library_path}: {error}") from error
    arguments = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_float] + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    library.lightx2v_sol_forward.argtypes = arguments
    library.lightx2v_sol_forward.restype = ctypes.c_int
    library.lightx2v_sol_forward_debug.argtypes = [*arguments, ctypes.c_void_p]
    library.lightx2v_sol_forward_debug.restype = ctypes.c_int
    library.musaGetErrorString.argtypes = [ctypes.c_int]
    library.musaGetErrorString.restype = ctypes.c_char_p
    return library, metadata


def _build_info() -> dict:
    """Load the native library and return its build metadata."""
    return json.loads(json.dumps(_load_library()[1]))


@functools.lru_cache(maxsize=None)
def _validate_device(device_index: int) -> None:
    # Torchada may expose MUSA tensors under a CUDA device alias.
    capability = torch.musa.get_device_capability(device_index)
    if capability != (3, 1):
        raise ValueError(f"MUSA native SOL requires MP31 / S5000 (capability 3.1), got {capability} on device {device_index}")


def _validate_shape_limits(batch: int, tokens: int, heads: int, padded_blocks: int) -> None:
    if any(value <= 0 or value >= 2**31 for value in (batch, tokens, heads, padded_blocks)):
        raise ValueError("MUSA native SOL dimensions must fit positive int32")
    # Descriptor strides use int32, including products of dimensions.
    if tokens > 2**31 - 1 - (BLOCK_SIZE - 1) or max(tokens, padded_blocks) * heads * HEAD_DIM >= 2**31:
        raise ValueError("MUSA native SOL requires T * H * 128 and padded_blocks * H * 128 to fit int32 descriptor strides")


def _validate_alignment(tensors) -> None:
    # Contiguous views can have storage offsets that break BF16 alignment.
    for name, value in zip(("q", "k", "v", "kc", "vc"), tensors):
        if value.data_ptr() % 32:
            raise ValueError(f"MUSA native SOL requires a 32-byte aligned {name} pointer; clone the tensor to remove its unaligned storage offset")


def _validate_native_inputs(q, k, v, kc, vc, threshold, scale, sink_start_block, sink_end_block, has_sink):
    batch, tokens, heads = _validate_qkv(q, k, v)
    tensors = (kc, vc, threshold)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("MUSA SOL requires tensor kc, vc, and threshold")
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    if kc.ndim != 4 or kc.shape != vc.shape or kc.shape[0] != batch or kc.shape[2:] != (heads, HEAD_DIM) or kc.shape[1] < blocks or kc.shape[1] % BLOCK_SIZE:
        raise ValueError("MUSA native SOL requires kc/vc [B, padded_blocks, H, 128], padded to a multiple of 64")
    _validate_shape_limits(batch, tokens, heads, kc.shape[1])
    if threshold.shape != (batch, blocks, heads):
        raise ValueError("MUSA native SOL requires threshold [B, ceil(T / 64), H]")
    if any(value.dtype != torch.bfloat16 for value in tensors[:-1]) or threshold.dtype != torch.float32:
        raise TypeError("MUSA native SOL requires BF16 q/k/v/kc/vc and FP32 threshold")
    if any(value.device != q.device for value in tensors):
        raise ValueError("MUSA native SOL requires all tensors on the same MUSA device")
    if not all(value.is_contiguous() for value in tensors):
        raise ValueError("MUSA native SOL requires contiguous tensors")
    if torch.is_grad_enabled() and any(value.requires_grad for value in tensors):
        raise ValueError("MUSA native SOL is forward-only")
    if not math.isfinite(float(scale)):
        raise ValueError("MUSA native SOL scale must be finite")
    if type(has_sink) is not bool or any(type(value) is not int for value in (sink_start_block, sink_end_block)):
        raise TypeError("MUSA native SOL requires boolean has_sink and integer sink block bounds")
    if not 0 <= sink_start_block <= sink_end_block <= blocks:
        raise ValueError("MUSA native SOL sink block range must be within [0, ceil(T / 64)]")
    _validate_alignment((q, k, v, kc, vc))


def _launch_native(q, k, v, kc, vc, threshold, *, scale, sink_start_block, sink_end_block, has_sink, debug):
    _validate_native_inputs(q, k, v, kc, vc, threshold, scale, sink_start_block, sink_end_block, has_sink)
    _validate_device(q.device.index)
    library, _ = _load_library()
    batch, tokens, heads, _ = q.shape
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    with torch.musa.device(q.device.index):
        output = torch.empty_like(q)
        stream = torch.musa.current_stream(q.device.index)
        arguments = [value.data_ptr() for value in (q, k, v, kc, vc, threshold, output)]
        arguments.extend([batch, tokens, heads, kc.shape[1], float(scale), sink_start_block, sink_end_block, int(has_sink), stream.musa_stream])
        routes = None
        if debug:
            routes = torch.empty((batch, heads, blocks, blocks), device=q.device, dtype=torch.uint8)
            result = library.lightx2v_sol_forward_debug(*arguments, routes.data_ptr())
        else:
            result = library.lightx2v_sol_forward(*arguments)
        if result:
            detail = library.musaGetErrorString(result)
            message = detail.decode("utf-8", errors="replace") if detail else "unknown MUSA error"
            raise RuntimeError(f"MUSA native SOL launch failed: {message} (code {result})")
    return (output, routes) if debug else output


def _native_forward(q, k, v, kc, vc, threshold, *, scale, sink_start_block, sink_end_block, has_sink):
    """Run forward with prepared KV summaries and thresholds."""
    return _launch_native(q, k, v, kc, vc, threshold, scale=scale, sink_start_block=sink_start_block, sink_end_block=sink_end_block, has_sink=has_sink, debug=False)


def _native_forward_debug(q, k, v, kc, vc, threshold, *, scale, sink_start_block, sink_end_block, has_sink):
    """Return output and a [B, H, NQ, NKV] route mask for correctness checks."""
    return _launch_native(q, k, v, kc, vc, threshold, scale=scale, sink_start_block=sink_start_block, sink_end_block=sink_end_block, has_sink=has_sink, debug=True)


@triton.jit
def _reduce_kv_kernel(
    k,
    v,
    kc,
    vc,
    T,
    TP,
    NPAD,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    tokens = block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    valid = tokens < T
    offsets = ((batch * TP + tokens[:, None]).to(tl.int64) * H + head) * D + dims[None, :]
    k_values = tl.load(k + offsets, mask=valid[:, None], other=0.0)
    v_values = tl.load(v + offsets, mask=valid[:, None], other=0.0)
    block_len = tl.minimum(BLOCK, T - block * BLOCK).to(tl.float32)
    summary_offsets = ((batch * NPAD + block) * H + head) * D + dims
    tl.store(kc + summary_offsets, tl.sum(k_values, axis=0) / block_len)
    tl.store(vc + summary_offsets, tl.sum(v_values, axis=0))


@functools.lru_cache(maxsize=1)
def _get_kv_reducer():
    # Autotune initializes the driver, so create it on first use.
    return triton.autotune(
        configs=[triton.Config({}, num_warps=warps, num_stages=stages) for warps in (4, 8) for stages in (1, 2)],
        key=["T", "H"],
    )(_reduce_kv_kernel)


@triton.jit
def _reduce_kc_stats_kernel(
    kc,
    kc_mean,
    kc_var_diag,
    NPAD,
    H: tl.constexpr,
    N: tl.constexpr,
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
    for start in range(0, N, GROUP):
        block_indices = start + blocks
        valid = block_indices < N
        offsets = ((batch * NPAD + block_indices[:, None]) * H + head) * D + dims[None, :]
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


@triton.jit
def _diag_threshold_kernel(
    q,
    kc_mean,
    kc_var_diag,
    threshold,
    scale,
    T,
    TP,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TAU: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    tokens = q_block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    valid = tokens < T
    offsets = ((batch * TP + tokens[:, None]).to(tl.int64) * H + head) * D + dims[None, :]
    q_values = tl.load(q + offsets, mask=valid[:, None], other=0.0)
    q_len = tl.minimum(BLOCK, T - q_block * BLOCK).to(tl.float32)
    q_centroid = tl.sum(q_values.to(tl.float32), axis=0) / q_len
    mean_kc = tl.load(kc_mean + batch_head * D + dims)
    var_kc = tl.load(kc_var_diag + batch_head * D + dims)
    log2_scale = scale * 1.4426950408889634
    mean = tl.sum(q_centroid * mean_kc, axis=0) * log2_scale
    variance = tl.sum(
        q_centroid * q_centroid * var_kc,
        axis=0,
    ) * (log2_scale * log2_scale)
    std = tl.sqrt(tl.maximum(variance, 0.0) + 1.0e-6)
    tl.store(
        threshold + (batch * N + q_block) * H + head,
        mean + TAU * std,
    )


@triton.jit
def _pool_query_kernel(
    q,
    q_bar,
    T,
    TP,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    tokens = q_block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    valid = tokens < T
    offsets = ((batch * TP + tokens[:, None]).to(tl.int64) * H + head) * D + dims[None, :]
    values = tl.load(q + offsets, mask=valid[:, None], other=0.0)
    q_len = tl.minimum(BLOCK, T - q_block * BLOCK).to(tl.float32)
    centroid = tl.sum(values.to(tl.float32), axis=0) / q_len
    tl.store(q_bar + (batch_head * N + q_block) * D + dims, centroid)


@triton.jit
def _exact_fused_threshold_kernel(
    q_bar,
    kc_mean,
    kc_second_moment,
    threshold,
    scale,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TAU: tl.constexpr,
):
    row_tile, batch_head = tl.program_id(0), tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    valid_rows = rows < N
    q_centroid = tl.load(
        q_bar + (batch_head * N + rows[:, None]) * D + dims[None, :],
        mask=valid_rows[:, None],
        other=0.0,
    )
    mean_kc = tl.load(kc_mean + batch_head * D + dims)
    second_moment = tl.load(kc_second_moment + batch_head * D * D + dims[:, None] * D + dims[None, :])
    raw_mean = tl.sum(q_centroid.to(tl.float32) * mean_kc[None, :], axis=1)
    projected = tl.dot(q_centroid, second_moment, out_dtype=tl.float32)
    raw_second_moment = tl.sum(
        projected * q_centroid.to(tl.float32),
        axis=1,
    )
    log2_scale = scale * 1.4426950408889634
    mean = raw_mean * log2_scale
    variance = tl.maximum(
        raw_second_moment - raw_mean * raw_mean,
        0.0,
    ) * (log2_scale * log2_scale)
    result = mean + TAU * tl.sqrt(variance + 1.0e-6)
    batch, head = batch_head // H, batch_head % H
    tl.store(
        threshold + (batch * N + rows) * H + head,
        result,
        mask=valid_rows,
    )


def _reduce_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, padded_tokens, heads, head_dim = k.shape
    tokens = padded_tokens if tokens is None else int(tokens)
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    padded_blocks = triton.cdiv(blocks, SUMMARY_PAD) * SUMMARY_PAD
    kc = torch.zeros(
        (batch, padded_blocks, heads, head_dim),
        device=k.device,
        dtype=torch.bfloat16,
    )
    vc = torch.zeros_like(kc)
    _get_kv_reducer()[(blocks, batch * heads)](
        k,
        v,
        kc,
        vc,
        tokens,
        padded_tokens,
        padded_blocks,
        heads,
        head_dim,
        BLOCK_SIZE,
    )
    return kc, vc


def _compute_diag_threshold(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float,
    scale: float,
    tokens: int | None = None,
) -> torch.Tensor:
    batch, padded_tokens, heads, head_dim = q.shape
    tokens = padded_tokens if tokens is None else int(tokens)
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    batch_heads = batch * heads
    kc_mean = torch.empty(
        (batch_heads, head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    kc_var_diag = torch.empty_like(kc_mean)
    threshold = torch.empty(
        (batch, blocks, heads),
        device=q.device,
        dtype=torch.float32,
    )
    _reduce_kc_stats_kernel[(batch_heads,)](
        kc,
        kc_mean,
        kc_var_diag,
        kc.shape[1],
        heads,
        blocks,
        head_dim,
        THRESHOLD_GROUP_SIZE,
        num_warps=4,
        num_stages=2,
    )
    _diag_threshold_kernel[(blocks, batch_heads)](
        q,
        kc_mean,
        kc_var_diag,
        threshold,
        scale,
        tokens,
        padded_tokens,
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tau,
        num_warps=4,
        num_stages=2,
    )
    return threshold


def _compute_exact_threshold(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float,
    scale: float,
    tokens: int | None = None,
) -> torch.Tensor:
    batch, padded_tokens, heads, head_dim = q.shape
    tokens = padded_tokens if tokens is None else int(tokens)
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    batch_heads = batch * heads
    kc_bh = kc[:, :blocks].permute(0, 2, 1, 3)
    kc_mean = kc_bh.mean(dim=2, dtype=torch.float32)
    kc_second_moment = torch.matmul(
        kc_bh.transpose(-1, -2),
        kc_bh,
    )
    kc_second_moment.div_(blocks)
    q_bar = torch.empty(
        (batch_heads, blocks, head_dim),
        device=q.device,
        dtype=torch.bfloat16,
    )
    threshold = torch.empty(
        (batch, blocks, heads),
        device=q.device,
        dtype=torch.float32,
    )
    _pool_query_kernel[(blocks, batch_heads)](
        q,
        q_bar,
        tokens,
        padded_tokens,
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    block_m = 64
    _exact_fused_threshold_kernel[(triton.cdiv(blocks, block_m), batch_heads)](
        q_bar,
        kc_mean,
        kc_second_moment,
        threshold,
        scale,
        heads,
        blocks,
        head_dim,
        block_m,
        tau,
        num_warps=4,
        num_stages=1,
    )
    return threshold


def _prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float,
    thresh_type: str = "diag",
    tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kc, vc = _reduce_kv(k, v, tokens=tokens)
    if thresh_type == "exact":
        threshold = _compute_exact_threshold(
            q,
            kc,
            tau=tau,
            scale=scale,
            tokens=tokens,
        )
    else:
        threshold = _compute_diag_threshold(
            q,
            kc,
            tau=tau,
            scale=scale,
            tokens=tokens,
        )
    return kc, vc, threshold


def _sink_block_range(tokens, sink_start, sink_tokens):
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    if sink_tokens == 0:
        return blocks, blocks
    start = tokens - sink_tokens if sink_start is None else sink_start
    return start // BLOCK_SIZE, (start + sink_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE


def sol_attn(
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
    """SOL self-attention for contiguous BF16 [B, T, H, 128] MUSA tensors.

    ``exact`` selects full-covariance thresholds, not dense attention.
    Sink tokens force exact evaluation of their KV blocks.
    """
    _validate_inputs(q, k, v, thresh_type=thresh_type, kv_splits=kv_splits, sink_tokens=sink_tokens, sink_start=sink_start)
    scale = HEAD_DIM**-0.5 if scale is None else float(scale)
    tau = float(tau)
    if not math.isfinite(scale) or not math.isfinite(tau) or tau < 0:
        raise ValueError("MUSA SOL attention scale must be finite and tau must be finite and non-negative")

    sink_start_block, sink_end_block = _sink_block_range(q.shape[1], sink_start, sink_tokens)
    with torch.musa.device(q.device.index):
        kc, vc, threshold = _prepare(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type)
        return _native_forward(q, k, v, kc, vc, threshold, scale=scale, sink_start_block=sink_start_block, sink_end_block=sink_end_block, has_sink=sink_tokens > 0)


__all__ = ["sol_attn"]
