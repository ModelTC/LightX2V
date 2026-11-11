"""ctypes bindings for HIP IPC memory, 1D async copies (SDMA for IPC peers) and stream write/wait-value ops."""

import ctypes

_hip = None
MEMCPY_D2D = 3
IPC_LAZY_PEER = 1
FINEGRAINED = 0x1
WAIT_GTE = 0


class IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def lib():
    global _hip
    if _hip is None:
        h = ctypes.CDLL("libamdhip64.so")
        vp, sz, u32 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32
        sig = {
            "hipMalloc": [ctypes.POINTER(vp), sz],
            "hipExtMallocWithFlags": [ctypes.POINTER(vp), sz, ctypes.c_uint],
            "hipFree": [vp],
            "hipIpcGetMemHandle": [ctypes.POINTER(IpcHandle), vp],
            "hipIpcOpenMemHandle": [ctypes.POINTER(vp), IpcHandle, ctypes.c_uint],
            "hipMemGetAddressRange": [ctypes.POINTER(vp), ctypes.POINTER(sz), vp],
            "hipMemcpyAsync": [vp, vp, sz, ctypes.c_int, vp],
            "hipMemsetAsync": [vp, ctypes.c_int, sz, vp],
            "hipMemset": [vp, ctypes.c_int, sz],
            "hipStreamWriteValue32": [vp, vp, u32, ctypes.c_uint],
            "hipStreamWaitValue32": [vp, vp, u32, ctypes.c_uint, u32],
        }
        for name, args in sig.items():
            fn = getattr(h, name)
            fn.argtypes, fn.restype = args, ctypes.c_int
        _hip = h
    return _hip


def check(status, what):
    if status:
        raise RuntimeError(f"{what} failed with hipError {status}")


def malloc(nbytes, flags=None):
    ptr = ctypes.c_void_p()
    if flags is None:
        check(lib().hipMalloc(ctypes.byref(ptr), nbytes), "hipMalloc")
    else:
        check(lib().hipExtMallocWithFlags(ctypes.byref(ptr), nbytes, flags), "hipExtMallocWithFlags")
    return ptr.value


def ipc_handle(ptr):
    """-> (handle bytes, offset of ptr inside its allocation)."""
    base, size, handle = ctypes.c_void_p(), ctypes.c_size_t(), IpcHandle()
    check(lib().hipMemGetAddressRange(ctypes.byref(base), ctypes.byref(size), ctypes.c_void_p(ptr)), "hipMemGetAddressRange")
    check(lib().hipIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(base.value)), "hipIpcGetMemHandle")
    return ctypes.string_at(ctypes.addressof(handle), 64), ptr - base.value


def ipc_open(handle_bytes, offset):
    handle, ptr = IpcHandle(), ctypes.c_void_p()
    ctypes.memmove(ctypes.addressof(handle), handle_bytes, 64)
    check(lib().hipIpcOpenMemHandle(ctypes.byref(ptr), handle, IPC_LAZY_PEER), "hipIpcOpenMemHandle")
    return ptr.value + offset


def copy(dst, src, nbytes, stream):
    check(lib().hipMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, MEMCPY_D2D, ctypes.c_void_p(stream)), "hipMemcpyAsync")


def write_value(stream, ptr, value):
    check(lib().hipStreamWriteValue32(ctypes.c_void_p(stream), ctypes.c_void_p(ptr), value, 0), "hipStreamWriteValue32")


def wait_value(stream, ptr, value):
    check(lib().hipStreamWaitValue32(ctypes.c_void_p(stream), ctypes.c_void_p(ptr), value, WAIT_GTE, 0xFFFFFFFF), "hipStreamWaitValue32")


class _Cai:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {"data": (ptr, False), "shape": (nbytes,), "typestr": "|u1", "version": 3}


def tensor(ptr, nbytes, device):
    """Wrap raw device memory (e.g. hipMalloc, IPC-shareable unlike expandable-segment tensors) as a uint8 tensor."""
    import torch

    return torch.as_tensor(_Cai(ptr, nbytes), device=device)
