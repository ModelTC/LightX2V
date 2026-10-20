import os
from functools import cache, lru_cache
from pathlib import Path
from threading import Lock

import torch
import torch.nn.functional as F

_launch_lock = Lock()


@cache
def _kernels(k, n, metallib_dir):
    if not metallib_dir:
        raise RuntimeError("int8-metal requires LIGHTX2V_METALLIB_DIR pointing to precompiled .metallib files")
    library = Path(metallib_dir).expanduser() / f"int8_k{k}_n{n}.metallib"
    if not library.is_file():
        raise FileNotFoundError(f"Missing precompiled INT8 Metal kernel for K={k}, N={n}: {library}")
    return torch.mps.load_metallib(library)


@lru_cache(maxsize=4)
def _workspace(m, k):
    return torch.empty(m, k, device="mps", dtype=torch.int8), torch.empty(m, device="mps", dtype=torch.float32)


def int8_matmul(x, weight, scale, bias=None):
    """X[M,K] @ INT8 W[K,N] -> a fresh BF16 output; W strides are (1,K).

    Requires LIGHTX2V_METALLIB_DIR with a precompiled library for (K,N).
    Libraries are cached per (K,N,directory); M stays dynamic.
    K must be a multiple of 1024, N of 128. Short/partial M is padded to 128.
    Scratch buffers are bounded and contain no weights or returned outputs.
    """
    if x.device.type != "mps" or x.dtype != torch.bfloat16 or x.ndim != 2 or x.requires_grad:
        raise ValueError("int8-metal requires an MPS BF16 matrix without autograd")
    if weight.device != x.device or weight.dtype != torch.int8 or weight.ndim != 2:
        raise ValueError("int8-metal requires MPS INT8 W[K,N]")
    m, k = x.shape
    n = weight.shape[1]
    if k <= 0 or n <= 0 or weight.shape[0] != k or k % 1024 or n % 128 or weight.stride() != (1, k):
        raise ValueError("int8-metal requires W[K,N] with strides (1,K), K divisible by 1024 and N by 128")
    mp = (m + 127) // 128 * 128
    if k * 127 * 128 >= 2**31 or max(mp * k, mp * n, k * n) >= 2**32:
        raise ValueError("int8-metal dimensions exceed INT32 accumulation or index limits")
    if scale.device != x.device or scale.dtype != torch.float32 or scale.numel() != n or not scale.is_contiguous():
        raise ValueError("int8-metal requires contiguous MPS FP32 scales with N elements")
    if bias is not None:
        if bias.device != x.device or bias.shape != (n,):
            raise ValueError("int8-metal bias must be an MPS vector with N elements")
        bias = bias.to(torch.bfloat16).contiguous()
    if not m:
        return torch.empty(0, n, device=x.device, dtype=x.dtype)
    if mp != m:
        x = F.pad(x, (0, 0, 0, mp - m))
    elif not x.is_contiguous() or x.storage_offset() % 4:
        x = x.clone(memory_format=torch.contiguous_format)
    output = torch.empty(mp, n, device=x.device, dtype=x.dtype)
    kernels = _kernels(k, n, os.environ.get("LIGHTX2V_METALLIB_DIR"))
    bn = 64 if n >= 8192 and k <= 4096 else 128
    nt = 256 if k % 2048 else 512
    # Keep the two enqueues together when multiple Python threads share MPS.
    with _launch_lock:
        xq, xs = _workspace(mp, k)
        kernels.quantize_x(x, xq, xs, threads=mp * nt, group_size=nt)
        getattr(kernels, f"gemm_{bn}")(
            xq,
            weight,
            xs,
            scale,
            output,
            bias if bias is not None else output,
            (mp, int(bias is not None)),
            threads=((n // bn) * 256, mp // 128),
            group_size=(256, 1),
            arg_casts="int32",
        )
    return output if mp == m else output[:m]
