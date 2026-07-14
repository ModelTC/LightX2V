import math

import torch
from packaging.version import parse

_KV_TORCH_VER = None


def _kvcache_dma_stream_priority() -> int:
    """Match WeightAsyncStreamManager cuda_load_stream priority."""
    global _KV_TORCH_VER
    if not torch.cuda.is_available():
        return 0
    if _KV_TORCH_VER is None:
        _KV_TORCH_VER = parse(torch.__version__.split("+")[0])
    return 1 if _KV_TORCH_VER >= parse("2.7") else 0


def cdiv(n: int, m: int) -> int:
    return (n + m - 1) // m


def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return max(a, b) or 1
    return a * b // math.gcd(a, b)
