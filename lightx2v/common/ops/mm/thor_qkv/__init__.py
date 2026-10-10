"""DreamZero Thor QKV projection using the thor-gemm-bench kernel."""
import ctypes
from functools import lru_cache
from pathlib import Path
from threading import local

import torch
from torch.utils.cpp_extension import load

_state = local()


@lru_cache(None)
def extension():
    root = Path(__file__).parent
    path = load(name="dreamzero_qkv_cutlass_v1", sources=[str(root / "adapter.cu")],
                extra_include_paths=["/opt/cutlass/include"],
                extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_110a", "--expt-relaxed-constexpr"],
                is_python_module=False, verbose=False)
    lib = ctypes.CDLL(path)
    lib.create.argtypes = [ctypes.c_void_p] * 5
    lib.create.restype = ctypes.c_void_p
    lib.launch_with.argtypes = [ctypes.c_void_p] * 6
    lib.launch_with.restype = ctypes.c_int
    lib.destroy.argtypes = [ctypes.c_void_p]
    return lib


class CutlassPlan:
    def __init__(self, lib, x, weight, bias, output):
        self.lib = lib
        self.device = x.device
        self.handle = lib.create(x.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
                                 torch.cuda.current_stream(x.device).cuda_stream)
        if not self.handle:
            raise RuntimeError("QKV CUTLASS initialization failed")

    def __del__(self):
        if getattr(self, "handle", None):
            try:
                with torch.cuda.device(self.device):
                    torch.cuda.synchronize(self.device)
                    self.lib.destroy(self.handle)
            except Exception:
                pass


def supports(modules, x):
    from lightx2v.common.ops.mm.mm_weight import MMWeight
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.shape == (1785, 5120)
            and x.is_contiguous() and torch.cuda.get_device_capability(x.device) == (11, 0)
            and all(type(m) is MMWeight and m._get_actual_bias() is not None
                    and m.weight.shape == (5120, 5120) and m.weight.stride() == (1, 5120)
                    and m.weight.dtype == x.dtype and m.weight.device == x.device
                    and not m.has_lora_branch for m in modules))


@torch.no_grad()
def project_qkv(modules, x):
    if not supports(modules, x):
        return tuple(m.apply(x) for m in modules)
    weights = tuple(m._get_actual_weight() for m in modules)
    biases = tuple(m._get_actual_bias() for m in modules)
    outputs = tuple(torch.empty_like(x) for _ in modules)
    lib = extension()
    key = (x.device, torch.cuda.current_stream(x.device).cuda_stream)
    cache = getattr(_state, "cache", None)
    if cache is None:
        cache = _state.cache = {}
    with torch.cuda.device(x.device):
        if key not in cache:
            cache[key] = CutlassPlan(lib, x, weights[0], biases[0], outputs[0])
        for weight, bias, output in zip(weights, biases, outputs):
            status = lib.launch_with(cache[key].handle, x.data_ptr(), weight.data_ptr(), bias.data_ptr(),
                                     output.data_ptr(), key[1])
            if status:
                raise RuntimeError(f"QKV CUTLASS launch failed: {status}")
    return outputs
