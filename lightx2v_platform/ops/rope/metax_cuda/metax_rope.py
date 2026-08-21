from functools import lru_cache
from importlib import import_module

import torch

from lightx2v_platform.ops.rope.rope_template import RopeTemplate
from lightx2v_platform.registry_factory import PLATFORM_ROPE_REGISTER


@lru_cache(maxsize=1)
def _load_metax_rotary_embedding_op():
    import_module("mcoplib._C")
    return torch.ops._C.rotary_embedding


@PLATFORM_ROPE_REGISTER("metax_rope")
class MetaxRope(RopeTemplate):
    def __init__(self, layout="interleaved", compute_dtype=torch.float32):
        super().__init__(layout=layout, compute_dtype=compute_dtype)
        _load_metax_rotary_embedding_op()

    def prepare_freqs(self, freqs, rotary_dim=None):
        if rotary_dim is not None and (rotary_dim <= 0 or rotary_dim % 2):
            raise ValueError(f"rotary_dim must be a positive even integer, got {rotary_dim}")

        if torch.is_tensor(freqs):
            freqs = freqs.reshape(-1, freqs.shape[-1])
            if torch.is_complex(freqs):
                freqs = torch.cat((freqs.real, freqs.imag), dim=-1)
        else:
            if rotary_dim is None:
                raise ValueError("rotary_dim is required for tuple RoPE frequencies")
            cos, sin = (cache.reshape(-1, cache.shape[-1]) for cache in freqs)
            if cos.shape != sin.shape:
                raise ValueError(f"RoPE cos/sin shapes must match, got cos={cos.shape}, sin={sin.shape}")
            if cos.shape[-1] == rotary_dim:
                if self.layout == "interleaved":
                    cos, sin = cos[..., ::2], sin[..., ::2]
                else:
                    cos, sin = cos[..., : rotary_dim // 2], sin[..., : rotary_dim // 2]
            freqs = torch.cat((cos, sin), dim=-1)

        if rotary_dim is not None and freqs.shape[-1] != rotary_dim:
            raise ValueError(f"RoPE cache width must be {rotary_dim}, got {freqs.shape[-1]}")
        return freqs.to(self.compute_dtype).contiguous()

    def prepare_positions(self, freqs):
        return torch.arange(freqs.shape[0], device=freqs.device, dtype=torch.long)

    def _apply(self, query, key, freqs, positions=None, rotary_dim=None):
        if rotary_dim is None and (not torch.is_tensor(freqs) or torch.is_complex(freqs)):
            rotary_dim = query.shape[-1]
        freqs = self.prepare_freqs(freqs, rotary_dim)

        if key is not None and (query.shape[:-2] != key.shape[:-2] or query.shape[-1] != key.shape[-1]):
            raise ValueError(f"query and key must have matching token dimensions and head size, got q={query.shape}, k={key.shape}")

        query_heads, head_size = query.shape[-2:]
        rotary_dim = freqs.shape[-1]
        if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > head_size:
            raise ValueError(f"rotary_dim must be positive, even, and <= head_size, got rotary_dim={rotary_dim}, head_size={head_size}")

        flat_query = query.reshape(-1, query_heads * head_size).contiguous()
        flat_key = None if key is None else key.reshape(-1, key.shape[-2] * head_size).contiguous()
        token_count = flat_query.shape[0]
        if token_count == 0:
            return query if key is None else (query, key)
        sequence_length = query.shape[-3]
        batch_size = token_count // sequence_length

        if positions is None:
            if freqs.shape[0] < sequence_length:
                raise ValueError(f"RoPE cache has {freqs.shape[0]} positions, expected at least {sequence_length}")
            positions = torch.arange(sequence_length, device=query.device, dtype=torch.long).repeat(batch_size)
        else:
            positions = positions.reshape(-1)
            if positions.shape[0] == sequence_length and batch_size > 1:
                positions = positions.repeat(batch_size)
            if positions.shape[0] != token_count:
                raise ValueError(f"positions must describe {token_count} tokens, got {positions.shape[0]}")
            positions = positions.to(device=query.device, dtype=torch.long).contiguous()

        torch.ops._C.rotary_embedding(
            positions,
            flat_query,
            flat_key,
            head_size,
            freqs,
            self.layout == "split_half",
        )
        query = flat_query.view_as(query)
        if key is None:
            return query
        return query, flat_key.view_as(key)

    def apply(self, query, key, freqs, positions=None, rotary_dim=None, **kwargs):
        return self._apply(query, key, freqs, positions, rotary_dim)

    def apply_single(self, tensor, freqs, positions=None, rotary_dim=None, **kwargs):
        return self._apply(tensor, None, freqs, positions, rotary_dim)


__all__ = ["MetaxRope"]
