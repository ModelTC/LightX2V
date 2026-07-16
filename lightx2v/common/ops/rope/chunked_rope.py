from __future__ import annotations

import torch

from lightx2v.utils.registry_factory import ROPE_REGISTER

from .template import RopeTemplate


def _slice_freqs(freqs, start: int, end: int):
    if isinstance(freqs, tuple):
        return tuple(item[start:end] for item in freqs)
    return freqs[start:end]


@ROPE_REGISTER("chunked_rope")
class ChunkedRope(RopeTemplate):
    def __init__(self, inner: RopeTemplate, chunk_size: int):
        super().__init__(layout=getattr(inner, "layout", "interleaved"), compute_dtype=getattr(inner, "compute_dtype", torch.float32))
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        self.inner = inner
        self.chunk_size = chunk_size

    def set_config(self, config=None):
        super().set_config(config)
        if hasattr(self.inner, "set_config"):
            self.inner.set_config(config)

    def apply_single(self, x: torch.Tensor, freqs, **kwargs):
        freq_len = freqs[0].shape[0] if isinstance(freqs, tuple) else freqs.shape[0]
        seq_len = min(x.shape[0], freq_len)
        output = torch.empty_like(x)
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            output[start:end] = self.inner.apply_single(x[start:end], _slice_freqs(freqs, start, end), **kwargs)
        if seq_len < x.shape[0]:
            output[seq_len:] = x[seq_len:]
        return output

    def apply(self, q: torch.Tensor, k: torch.Tensor, freqs, **kwargs):
        freq_len = freqs[0].shape[0] if isinstance(freqs, tuple) else freqs.shape[0]
        seq_len = min(q.shape[0], k.shape[0], freq_len)
        q_out, k_out = torch.empty_like(q), torch.empty_like(k)
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            q_chunk, k_chunk = self.inner.apply(q[start:end], k[start:end], _slice_freqs(freqs, start, end), **kwargs)
            q_out[start:end], k_out[start:end] = q_chunk, k_chunk
        if seq_len < q.shape[0]:
            q_out[seq_len:], k_out[seq_len:] = q[seq_len:], k[seq_len:]
        return q_out, k_out
