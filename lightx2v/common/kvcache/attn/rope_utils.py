"""Normalize model RoPE tables while keeping stable graph-visible storage."""

import torch


class RotaryFrequencyCache:
    """One mutable frequency workspace per output layout.

    Consume a prepared table before preparing different contents for that
    layout. Captured graphs read the workspace refreshed for the current call.
    """

    def __init__(self):
        self.buffers = {}
        self._revision = 0
        self.last_revision = 0

    def prepare(self, freqs, rotary_dim, layout):
        tensors = freqs if isinstance(freqs, tuple) else (freqs,)
        signature = tuple((x.data_ptr(), x.shape, x.stride(), x.dtype, x.device) for x in tensors), rotary_dim, layout
        rows = tensors[0].numel() // tensors[0].shape[-1]
        storage_key = rows, rotary_dim, layout, tensors[0].device
        versions = []
        for tensor in tensors:
            try:
                versions.append(tensor._version)
            except RuntimeError:
                # Inference tensors have no version counter. Re-read those
                # tables on each preparation while retaining output storage.
                versions.append(None)
        versions = tuple(versions)
        cached = self.buffers.get(storage_key)
        if cached is not None:
            _, cached_signature, cached_versions, prepared, revision = cached
            if signature == cached_signature and None not in versions and cached_versions == versions:
                self.last_revision = revision
                return prepared
        if isinstance(freqs, tuple):
            cosine, sine = freqs
            if cosine.shape != sine.shape:
                raise ValueError("RoPE cosine and sine tables must have identical shapes")
            if cosine.shape[-1] == rotary_dim:
                selector = slice(None, None, 2) if layout == "interleaved" else slice(0, rotary_dim // 2)
                cosine, sine = cosine[..., selector], sine[..., selector]
            if cosine.shape[-1] != rotary_dim // 2:
                raise ValueError("RoPE tuple width must be rotary_dim or rotary_dim / 2")
            parts = cosine, sine
        elif freqs.is_complex():
            parts = freqs.real, freqs.imag
        else:
            parts = (freqs,)
        if sum(part.shape[-1] for part in parts) != rotary_dim:
            raise ValueError("RoPE table must contain cosine | sine with width rotary_dim")
        if cached is None:
            prepared = torch.empty((rows, rotary_dim), device=parts[0].device, dtype=torch.float32)
        if len(parts) == 1:
            prepared.copy_(parts[0].reshape(-1, rotary_dim))
        else:
            torch.cat(tuple(part.reshape(-1, rotary_dim // 2) for part in parts), dim=-1, out=prepared)
        self._revision += 1
        self.last_revision = self._revision
        # Bind graphs to a shape's output storage; source tables may be newly
        # allocated for every request. Retain only the latest source per shape.
        self.buffers[storage_key] = tensors, signature, versions, prepared, self.last_revision
        return prepared

    def clear(self):
        self.buffers.clear()


def apply_model_rope(rope, tensor, freqs, rotary_dim):
    if rotary_dim != tensor.shape[-1]:
        return rope.apply_single(tensor, freqs, rotary_dim=rotary_dim)
    return rope.apply_single(tensor, freqs)
