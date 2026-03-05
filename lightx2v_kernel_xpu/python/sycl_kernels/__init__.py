# isort: skip_file
import ctypes
import os

_pkg_dir = os.path.dirname(os.path.abspath(__file__))

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

from sycl_kernels._ext import sdp, onednn_w4a16, onednn_w8a16_fp8  # noqa: E402, F401
from sycl_kernels.version import __version__  # noqa: E402, F401
