"""JIT build of the gfx1201 SageAttention HIP library.

The kernels are compiled once with ``hipcc`` into ``libhip_sage.so`` and cached under
``$LIGHTX2V_HIP_SAGE_CACHE`` (default ``~/.cache/lightx2v/hip_sage``), keyed by the sources, the compiler
and the target. Ranks of a distributed job share the cache through a file lock.
"""

import ctypes
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache

from loguru import logger

ARCH = "gfx1201"
_CSRC = os.path.join(os.path.dirname(__file__), "csrc")
_SOURCES = ("sage_prep.hip", "sage_attn_int4.hip", "sage_attn_int8.hip")
_DEPS = ("sage_common.h",)
_FLAGS = ("-O3", "-std=c++17", f"--offload-arch={ARCH}", "-shared", "-fPIC")
# Older HIP compilers allocate far more VGPRs for these kernels (lower occupancy, spills) and run them ~10% slower.
_RECOMMENDED_HIP = (7, 15)
_ENTRY_POINTS = ("hip_sage_kv_stats", "hip_sage_q_prep", "hip_sage_k_prep", "hip_sage_v_prep", "hip_sage_ds", "hip_sage_attn")


def _hipcc():
    rocm = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH") or "/opt/rocm"
    candidate = os.path.join(rocm, "bin", "hipcc")
    path = candidate if os.path.exists(candidate) else shutil.which("hipcc")
    if path is None:
        raise RuntimeError("hip_sage: hipcc not found; set ROCM_PATH or put hipcc on PATH")
    return path


def _hip_version(hipcc):
    out = subprocess.run([hipcc, "--version"], capture_output=True, text=True, check=True).stdout
    m = re.search(r"HIP version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None, out


def _build(hipcc, target):
    srcs = [os.path.join(_CSRC, s) for s in _SOURCES]
    with tempfile.TemporaryDirectory(dir=os.path.dirname(target)) as tmp:
        tmp_so = os.path.join(tmp, os.path.basename(target))
        cmd = [hipcc, *_FLAGS, "-o", tmp_so, *srcs]
        logger.info(f"[hip_sage] building {os.path.basename(target)}: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"hip_sage: hipcc failed ({res.returncode}):\n{res.stderr[-4000:]}")
        os.replace(tmp_so, target)


@lru_cache(maxsize=1)
def load_library():
    """Build (if needed) and load libhip_sage.so; returns the ctypes handle."""
    hipcc = _hipcc()
    version, version_text = _hip_version(hipcc)
    if version is not None and version < _RECOMMENDED_HIP:
        logger.warning(f"[hip_sage] HIP {version[0]}.{version[1]} detected; HIP >= {_RECOMMENDED_HIP[0]}.{_RECOMMENDED_HIP[1]} is recommended for these kernels")
    digest = hashlib.sha256()
    for name in (*_SOURCES, *_DEPS):
        with open(os.path.join(_CSRC, name), "rb") as f:
            digest.update(f.read())
    digest.update(" ".join(_FLAGS).encode())
    digest.update(version_text.encode())
    cache = os.path.expanduser(os.environ.get("LIGHTX2V_HIP_SAGE_CACHE", "~/.cache/lightx2v/hip_sage"))
    os.makedirs(cache, exist_ok=True)
    target = os.path.join(cache, f"libhip_sage_{ARCH}_{digest.hexdigest()[:16]}.so")
    if not os.path.exists(target):
        with open(target + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if not os.path.exists(target):
                    _build(hipcc, target)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    lib = ctypes.CDLL(target)
    for fn in _ENTRY_POINTS:
        getattr(lib, fn).restype = ctypes.c_int
    return lib
