import ctypes
import os
import platform
from lightx2v_kernel import common_ops  # noqa: F401
from lightx2v_kernel.version import __version__
from lightx2v_kernel.svdq import (  # noqa: F401
    svdq_gemm_w4a4_cuda,
    svdq_quantize_w4a4_act_fuse_lora_cuda,
)


SYSTEM_ARCH = platform.machine()

cuda_path = f"/usr/local/cuda/targets/{SYSTEM_ARCH}-linux/lib/libcudart.so.12"
if os.path.exists(cuda_path):
    ctypes.CDLL(cuda_path, mode=ctypes.RTLD_GLOBAL)


build_tree_kernel = None
