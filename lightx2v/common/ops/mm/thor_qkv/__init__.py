"""DreamZero Thor BF16 projections using the thor-gemm-bench CUTLASS kernel.

``linear`` computes ``x @ W + b`` for one MMWeight; ``project_qkv`` applies
several MMWeights to the same input.  The CUTLASS kernel (tile 256x256x64,
2-SM cluster 2x1, fused BF16 bias) accumulates K in order without split-K and
is bitwise identical to the cuBLAS ``addmm`` path it replaces for any number of
rows; unsupported inputs fall back to ``MMWeight.apply``.
"""

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(None)
def extension():
    root = Path(__file__).parent
    # CUTLASS_PATH points to the checkout root, not its include directory.
    cutlass_root = Path(os.environ.get("CUTLASS_PATH") or "/opt/cutlass").expanduser().resolve()
    path = load(
        name="dreamzero_thor_linear_cutlass_v2",
        sources=[str(root / "adapter.cu")],
        extra_include_paths=[str(cutlass_root / "include")],
        # -DNDEBUG: CUTLASS device asserts otherwise stay in the kernel.
        extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_110a", "--expt-relaxed-constexpr", "-DNDEBUG"],
        is_python_module=False,
        verbose=False,
    )
    lib = ctypes.CDLL(path)
    lib.thor_linear_bias.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_longlong, ctypes.c_void_p]
    lib.thor_linear_bias.restype = ctypes.c_int
    return lib


def _aligned(t):
    # TMA descriptors need 16-byte aligned base addresses; misaligned views fall back.
    return t.data_ptr() % 16 == 0


def _module_ok(m, x):
    from lightx2v.common.ops.mm.mm_weight import MMWeight

    if type(m) is not MMWeight or m.has_lora_branch:
        return False
    w, b = m._get_actual_weight(), m._get_actual_bias()
    # MMWeight keeps W as the transposed view of the checkpoint's [out, in] tensor.
    return (
        b is not None
        and _aligned(w)
        and _aligned(b)
        and w.dim() == 2
        and w.shape[0] == x.shape[1]
        and w.stride() == (1, w.shape[0])
        and w.shape[1] % 8 == 0
        and w.shape[0] % 8 == 0
        and w.dtype == x.dtype
        and w.device == x.device
        and b.shape == (w.shape[1],)
        and b.is_contiguous()
        and b.dtype == x.dtype
        and b.device == x.device
    )


def _input_ok(x):
    return x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[0] > 0 and x.is_contiguous() and _aligned(x) and torch.cuda.get_device_capability(x.device) == (11, 0)


def _output_ok(o, x, n):
    return o.shape == (x.shape[0], n) and o.dtype == x.dtype and o.device == x.device and o.is_contiguous() and _aligned(o)


def supports(module, x):
    """True if ``linear(module, x)`` runs the CUTLASS kernel (else it falls back)."""
    return _input_ok(x) and _module_ok(module, x)


def _launch(lib, m, x, out):
    w, b = m._get_actual_weight(), m._get_actual_bias()
    rows, k = x.shape
    n = w.shape[1]
    status = lib.thor_linear_bias(x.data_ptr(), w.data_ptr(), b.data_ptr(), out.data_ptr(), rows, n, k, n, torch.cuda.current_stream(x.device).cuda_stream)
    if status:
        raise RuntimeError(f"Thor CUTLASS linear failed: {status}")


def _fallback(m, x, out):
    y = m.apply(x)
    return y if out is None else out.copy_(y)


@torch.no_grad()
def linear(module, x, out=None):
    """``module.apply(x)`` (bias included), optionally written into ``out``."""
    if not supports(module, x):
        return _fallback(module, x, out)
    n = module._get_actual_weight().shape[1]
    if out is None:
        out = torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)
    elif not _output_ok(out, x, n):
        return _fallback(module, x, out)
    lib = extension()
    with torch.cuda.device(x.device):
        _launch(lib, module, x, out)
    return out


@torch.no_grad()
def project_qkv(modules, x, outputs=None):
    """Project ``x`` with each module.  ``outputs`` optionally gives, per module,
    a contiguous [tokens, out_features] tensor to write that projection into
    (None entries are allocated)."""
    if outputs is None:
        outputs = (None,) * len(modules)
    return tuple(linear(m, x, o) for m, o in zip(modules, outputs))
