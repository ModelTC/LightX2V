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

    def prepare_freqs(self, freqs, rotary_dim: int | None = None):
        return self.inner.prepare_freqs(freqs, rotary_dim=rotary_dim)

    def prepare_positions(self, freqs):
        return self.inner.prepare_positions(freqs)

    def apply_single(self, x: torch.Tensor, freqs, positions: torch.Tensor | None = None, **kwargs):
        freq_len = freqs[0].shape[0] if isinstance(freqs, tuple) else freqs.shape[0]
        seq_len = min(x.shape[0], freq_len)
        if positions is not None:
            seq_len = min(seq_len, positions.shape[0])
        output = torch.empty_like(x)
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk_freqs = freqs if positions is not None else _slice_freqs(freqs, start, end)
            chunk_positions = positions[start:end] if positions is not None else None
            if chunk_positions is None:
                output[start:end] = self.inner.apply_single(x[start:end], chunk_freqs, **kwargs)
            else:
                output[start:end] = self.inner.apply_single(
                    x[start:end],
                    chunk_freqs,
                    positions=chunk_positions,
                    **kwargs,
                )
        if seq_len < x.shape[0]:
            output[seq_len:] = x[seq_len:]
        return output

    def apply(self, q: torch.Tensor, k: torch.Tensor, freqs, positions: torch.Tensor | None = None, **kwargs):
        freq_len = freqs[0].shape[0] if isinstance(freqs, tuple) else freqs.shape[0]
        seq_len = min(q.shape[0], k.shape[0], freq_len)
        if positions is not None:
            seq_len = min(seq_len, positions.shape[0])
        q_out, k_out = torch.empty_like(q), torch.empty_like(k)
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk_freqs = freqs if positions is not None else _slice_freqs(freqs, start, end)
            chunk_positions = positions[start:end] if positions is not None else None
            if chunk_positions is None:
                q_chunk, k_chunk = self.inner.apply(q[start:end], k[start:end], chunk_freqs, **kwargs)
            else:
                q_chunk, k_chunk = self.inner.apply(
                    q[start:end],
                    k[start:end],
                    chunk_freqs,
                    positions=chunk_positions,
                    **kwargs,
                )
            q_out[start:end], k_out[start:end] = q_chunk, k_chunk
        if seq_len < q.shape[0]:
            q_out[seq_len:], k_out[seq_len:] = q[seq_len:], k[seq_len:]
        return q_out, k_out
