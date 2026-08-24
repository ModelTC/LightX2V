# isort: skip_file
import ctypes
import glob
import os

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_cute_fmha_loaded = False
_rms_norm_loaded = False
_minimax_h3_rope_loaded = False

if os.name == "nt":
    os.add_dll_directory(_pkg_dir)
    # Add torch's lib dir so _ext can find dnnl.dll and other torch-bundled DLLs
    # before torch itself is imported by the caller.
    try:
        import torch as _torch

        _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            os.add_dll_directory(_torch_lib)
        del _torch, _torch_lib
    except ImportError:
        pass
    _dll = os.path.join(_pkg_dir, "esimd.unify.lgrf.dll")
    if os.path.isfile(_dll):
        ctypes.CDLL(_dll)
    else:
        raise FileNotFoundError(f"esimd.unify.lgrf.dll not found in {_pkg_dir}")
else:
    # Load explicitly with global visibility so the extension can resolve the
    # ESIMD entry points even on distributions that default to local dlopen.
    _so = os.path.join(_pkg_dir, "libesimd.unify.lgrf.so")
    if os.path.isfile(_so):
        ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
    else:
        raise FileNotFoundError(f"libesimd.unify.lgrf.so not found in {_pkg_dir}")

try:
    from sycl_kernels._ext import (  # noqa: E402, F401
        onednn_w4a16,
        onednn_w8a8_int8,
        onednn_w8a16_fp8,
        sdp,
    )
except ImportError as _legacy_import_error:

    def _legacy_extension_unavailable(*args, _error=_legacy_import_error, **kwargs):
        raise RuntimeError("sycl_kernels legacy ESIMD/oneDNN extension could not be loaded; check that the oneDNN headers and libdnnl runtime have matching versions") from _error

    onednn_w4a16 = _legacy_extension_unavailable
    onednn_w8a8_int8 = _legacy_extension_unavailable
    onednn_w8a16_fp8 = _legacy_extension_unavailable
    sdp = _legacy_extension_unavailable
from sycl_kernels.version import __version__  # noqa: E402, F401


def _load_rms_norm():
    global _rms_norm_loaded
    if _rms_norm_loaded:
        return
    import torch

    suffix = "*.pyd" if os.name == "nt" else "*.so"
    candidates = sorted(glob.glob(os.path.join(_pkg_dir, "rms_norm_torch" + suffix)))
    if not candidates:
        raise ImportError(f"rms_norm_torch library not found in {_pkg_dir}")
    torch.ops.load_library(candidates[0])
    _rms_norm_loaded = True


def rms_norm(weight, input, eps=1e-6):
    """Run ESIMD RMSNorm on contiguous [rows, hidden_size] XPU tensors."""
    import torch

    try:
        op = torch.ops.sycl_kernels_rms.rms_norm
    except AttributeError:
        _load_rms_norm()
        op = torch.ops.sycl_kernels_rms.rms_norm
    return op(weight, input, eps)


def has_rms_norm():
    try:
        _load_rms_norm()
        return True
    except (ImportError, OSError, RuntimeError):
        return False


def _load_minimax_h3_rope():
    global _minimax_h3_rope_loaded
    if _minimax_h3_rope_loaded:
        return
    import torch

    suffix = "*.pyd" if os.name == "nt" else "*.so"
    candidates = sorted(glob.glob(os.path.join(_pkg_dir, "minimax_h3_rope_torch" + suffix)))
    if not candidates:
        raise ImportError(f"minimax_h3_rope_torch library not found in {_pkg_dir}")
    torch.ops.load_library(candidates[0])
    _minimax_h3_rope_loaded = True


def minimax_h3_rope(input, freqs):
    """Apply MiniMax-H3 partial split-half RoPE to BF16 [rows, heads, 128]."""
    import torch

    try:
        op = torch.ops.sycl_kernels_minimax_h3.rope
    except AttributeError:
        _load_minimax_h3_rope()
        op = torch.ops.sycl_kernels_minimax_h3.rope
    return op(input, freqs)


def minimax_h3_rope_cached(input, cos, sin):
    """Apply MiniMax-H3 RoPE using precomputed FP32 cosine and sine caches."""
    import torch

    try:
        op = torch.ops.sycl_kernels_minimax_h3.rope_cached
    except AttributeError:
        _load_minimax_h3_rope()
        op = torch.ops.sycl_kernels_minimax_h3.rope_cached
    return op(input, cos, sin)


def has_minimax_h3_rope():
    try:
        _load_minimax_h3_rope()
        return True
    except (ImportError, OSError, RuntimeError):
        return False


def _load_cute_fmha():
    global _cute_fmha_loaded

    if os.name == "nt":
        # The current sycl-tla FMHA kernel builds on Windows but produces
        # incorrect attention results, so do not allow it to be loaded.
        raise RuntimeError("CUTE FMHA is disabled on Windows because sycl-tla produces incorrect results")
    if _cute_fmha_loaded:
        return
    import torch

    candidates = sorted(glob.glob(os.path.join(_pkg_dir, "cute_fmha_torch*.so")))
    if not candidates:
        raise ImportError(f"cute_fmha_torch.so not found in {_pkg_dir}")
    torch.ops.load_library(candidates[0])
    _cute_fmha_loaded = True


def cute_sdp(q, k, v):
    """Run generic CUTLASS-SYCL CUTE self-attention on [B,L,H,128]."""
    import torch

    try:
        op = torch.ops.sycl_kernels_cute.sdp
    except AttributeError:
        _load_cute_fmha()
        op = torch.ops.sycl_kernels_cute.sdp
    return op(q, k, v)


def has_cute_fmha():
    if os.name == "nt":
        return False
    try:
        _load_cute_fmha()
        return True
    except (ImportError, OSError, RuntimeError):
        return False
