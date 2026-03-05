import ctypes
import os

from sycl_kernels._ext import onednn_w4a16, onednn_w8a16_fp8, sdp  # noqa: E402, F401
from sycl_kernels.version import __version__  # noqa: F401

_pkg_dir = os.path.dirname(os.path.abspath(__file__))

if os.name == "nt":
    os.add_dll_directory(_pkg_dir)
    _dll = os.path.join(_pkg_dir, "esimd.unify.lgrf.dll")
    if os.path.isfile(_dll):
        ctypes.CDLL(_dll)
    else:
        raise FileNotFoundError(f"esimd.unify.lgrf.dll not found in {_pkg_dir}")
