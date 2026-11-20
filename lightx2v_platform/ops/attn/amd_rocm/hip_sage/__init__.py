"""SageAttention for AMD gfx1201 (RDNA4): smoothed INT4 or INT8 Q.K^T, FP8 P.V, bf16 [N, H, 128] in/out.

``attention(q, k, v, softmax_scale, bits)`` quantizes Q/K/V, builds the Q-smoothing term and runs the
matching HIP kernel for ``bits`` in {4, 8}. Heads are processed in chunks of ``HEAD_CHUNK`` to bound the
size of the smoothing table (fp16 [chunk, N / 128, N]).
"""

import ctypes
import math

import torch

from .jit import ARCH, load_library

HD = 128
BQ = 128
BK = 32
HEAD_CHUNK = 4
LOG2E = 1.4426950408889634

__all__ = ["ARCH", "attention", "is_supported_device", "supports"]


def is_supported_device(device=None):
    if getattr(torch.version, "hip", None) is None or not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(device if device is not None else torch.cuda.current_device())
    return props.gcnArchName.split(":")[0] == ARCH


def supports(q, k, v):
    """Shapes and dtypes the kernels handle: [N, H <= 64, 128] bf16, one sequence, identical Q/K/V shapes."""
    return q.dim() == 3 and q.shape == k.shape == v.shape and q.shape[-1] == HD and q.shape[1] <= 64 and q.dtype == k.dtype == v.dtype == torch.bfloat16 and q.is_cuda


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


def _check(rc, name):
    if rc:
        raise RuntimeError(f"hip_sage {name} failed (rc={rc})")


def _kv_stats(lib, k, v, stream):
    """K column mean [H,128] fp32 and V fp8 scale amax|v|/448 [H,128]."""
    N, H, _ = k.shape
    ks = torch.zeros(H, HD, device=k.device, dtype=torch.float32)
    vm = torch.zeros(H, HD, device=k.device, dtype=torch.int32)
    _check(lib.hip_sage_kv_stats(_p(k), _p(v), N, H, _p(ks), _p(vm), stream), "kv_stats")
    return ks.div_(N), vm.view(torch.float32).clamp_min_(1e-20).div_(448)


def _prep(lib, q, k, v, h0, hc, scale, N, NqPad, NkPad, kmean, sv, stream, bits):
    """Kernel operands for heads [h0, h0+hc)."""
    H = q.shape[1]
    dev = q.device
    nqb = NqPad // BQ
    nqbPad = -(-nqb // 128) * 128
    pk = HD * bits // 8
    qh = torch.empty(hc, NqPad, pk, device=dev, dtype=torch.uint8)
    sq = torch.empty(hc, NqPad, device=dev, dtype=torch.float32)
    qbar = torch.empty(hc, nqb, HD, device=dev, dtype=torch.float32)
    _check(lib.hip_sage_q_prep(bits, _p(q), N, H, h0, hc, ctypes.c_float(scale), NqPad, _p(qh), _p(sq), _p(qbar), stream), "q_prep")
    kh = torch.empty(hc, NkPad, pk, device=dev, dtype=torch.uint8)
    sk = torch.empty(hc, NkPad // 8, device=dev, dtype=torch.float32)
    _check(lib.hip_sage_k_prep(bits, _p(k), _p(kmean), N, H, h0, hc, NkPad, _p(kh), _p(sk), stream), "k_prep")
    vt = torch.empty(hc, NkPad // BK, HD, BK, device=dev, dtype=torch.uint8)
    _check(lib.hip_sage_v_prep(_p(v), _p(sv), N, H, h0, hc, NkPad, _p(vt), stream), "v_prep")
    # k mean-centering shifts each dS row by a constant that the row shift cancels, so dS uses raw k.
    qs = torch.zeros(2, hc, nqbPad, HD, device=dev, dtype=torch.bfloat16)
    qs[0, :, :nqb] = qbar.bfloat16()
    qs[1, :, :nqb] = (qbar - qs[0, :, :nqb].float()).bfloat16()
    rkey = torch.empty(hc, nqbPad, device=dev, dtype=torch.int32)
    ds = torch.empty(hc, nqb, NkPad, device=dev, dtype=torch.float16)
    _check(lib.hip_sage_ds(_p(qs), _p(k), N, NkPad, H, h0, hc, nqb, nqbPad, _p(rkey), _p(ds), stream), "ds")
    return qh, sq, kh, sk, ds, vt, sv[h0 : h0 + hc].contiguous()


@torch.no_grad()
def attention(q, k, v, softmax_scale=None, bits=8):
    """Non-causal attention of one sequence. q, k, v: [N, H, 128] bf16. Returns [N, H, 128] bf16."""
    if bits not in (4, 8):
        raise ValueError(f"hip_sage supports bits=4 or 8, got {bits}")
    lib = load_library()
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    N, H, _ = q.shape
    scale = (softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(HD)) * LOG2E
    NqPad, NkPad = -(-N // 256) * 256, -(-N // 64) * 64
    stream = ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream)
    kmean, sv = _kv_stats(lib, k, v, stream)
    out = torch.empty(N, H, HD, device=q.device, dtype=torch.bfloat16)
    for h0 in range(0, H, HEAD_CHUNK):
        hc = min(HEAD_CHUNK, H - h0)
        ops = _prep(lib, q, k, v, h0, hc, scale, N, NqPad, NkPad, kmean, sv, stream, bits)
        _check(lib.hip_sage_attn(bits, *[_p(t) for t in ops], _p(out), N, NqPad, NkPad, H, h0, hc, stream), f"attn_int{bits}")
        del ops
    return out
