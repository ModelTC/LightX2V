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


def causal_chunk_token_range(chunk_index: int, num_chunks: int, total_tokens: int) -> tuple[int, int]:
    """Map a temporal chunk to token bounds using the teacher-forcing convention."""
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}.")
    if not 0 <= chunk_index < num_chunks:
        raise ValueError(f"chunk_index must be in [0, {num_chunks}), got {chunk_index}.")
    if total_tokens < num_chunks:
        raise ValueError(f"total_tokens={total_tokens} must be at least num_chunks={num_chunks}.")

    start = (chunk_index * total_tokens + num_chunks - 1) // num_chunks
    end = ((chunk_index + 1) * total_tokens + num_chunks - 1) // num_chunks
    return start, end
