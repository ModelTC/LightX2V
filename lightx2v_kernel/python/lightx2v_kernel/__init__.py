import ctypes
import os
import platform
from lightx2v_kernel import common_ops  # noqa: F401
from lightx2v_kernel.version import __version__


SYSTEM_ARCH = platform.machine()

cuda_path = f"/usr/local/cuda/targets/{SYSTEM_ARCH}-linux/lib/libcudart.so.12"
if os.path.exists(cuda_path):
    ctypes.CDLL(cuda_path, mode=ctypes.RTLD_GLOBAL)


build_tree_kernel = None
