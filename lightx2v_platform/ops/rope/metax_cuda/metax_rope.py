from functools import lru_cache
from importlib import import_module

import torch

from lightx2v_platform.ops.rope.rope_template import RopeTemplate
from lightx2v_platform.registry_factory import PLATFORM_ROPE_REGISTER


@lru_cache(maxsize=1)
def _load_metax_rotary_embedding_op():
    import_module("mcoplib._C")
    return torch.ops._C.rotary_embedding


@torch.library.custom_op(
    "lightx2v::metax_rotary_embedding",
    mutates_args=(),
    device_types="cuda",
)
def metax_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    head_size: int,
    split_half: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = query.contiguous().clone()
    key = key.contiguous().clone()
    flat_query = query.view(-1, query.shape[-2] * head_size)
    flat_key = key.view(-1, key.shape[-2] * head_size)
    _load_metax_rotary_embedding_op()(positions, flat_query, flat_key, head_size, freqs, split_half)
    return query, key


@metax_rotary_embedding.register_fake
def _metax_rotary_embedding_fake(query, key, positions, freqs, head_size, split_half):
    return query.new_empty(query.shape), key.new_empty(key.shape)


@torch.library.custom_op(
    "lightx2v::metax_rotary_embedding_single",
    mutates_args=(),
    device_types="cuda",
)
def metax_rotary_embedding_single(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    head_size: int,
    split_half: bool,
) -> torch.Tensor:
    output = tensor.contiguous().clone()
    flat_output = output.view(-1, output.shape[-2] * head_size)
    _load_metax_rotary_embedding_op()(positions, flat_output, None, head_size, freqs, split_half)
    return output


@metax_rotary_embedding_single.register_fake
def _metax_rotary_embedding_single_fake(tensor, positions, freqs, head_size, split_half):
    return tensor.new_empty(tensor.shape)


@lru_cache(maxsize=1)
def _register_metax_rotary_embedding_lowering():
    from torch._inductor.ir import Pointwise
    from torch._inductor.lowering import register_lowering
    from torch._inductor.virtualized import ops
    from torch.utils._sympy.functions import FloorDiv, ModularIndexing

    def rotate(tensor, positions, freqs, head_size, split_half):
        tensor_loader = tensor.make_loader()
        positions_loader = positions.make_loader()
        freqs_loader = freqs.make_loader()
        rotary_dim = freqs.get_size()[-1]
        half = FloorDiv(rotary_dim, 2)
        input_dtype = tensor.get_dtype()
        compute_dtype = freqs.get_dtype()
        tensor_size = tensor.get_size()

        def inner_fn(index):
            channel = index[-1]
            channel_value = ops.index_expr(channel, torch.int64)
            rotary_mask = ops.lt(channel_value, ops.index_expr(rotary_dim, torch.int64))

            if split_half:
                pair = ModularIndexing(channel, 1, half)
                partner_channel = ModularIndexing(channel + half, 1, rotary_dim)
                first = ops.lt(channel_value, ops.index_expr(half, torch.int64))
            else:
                parity = ModularIndexing(channel, 1, 2)
                pair = ModularIndexing(channel, 2, half)
                partner_channel = ModularIndexing(channel + 1 - 2 * parity, 1, head_size)
                first = ops.eq(ops.index_expr(parity, torch.int64), ops.constant(0, torch.int64))

            partner_index = list(index)
            partner_index[-1] = partner_channel
            token = 0
            for token_index, size in zip(index[:-2], tensor_size[:-2]):
                token = token * size + token_index
            position = positions_loader([token])
            row = ops.indirect_indexing(position, freqs.get_size()[0], check=True, wrap_neg=False)

            value = ops.to_dtype(tensor_loader(index), compute_dtype)
            partner = ops.to_dtype(tensor_loader(partner_index), compute_dtype)
            cos = ops.to_dtype(freqs_loader([row, pair]), compute_dtype)
            sin = ops.to_dtype(freqs_loader([row, pair + half]), compute_dtype)
            rotated = ops.where(first, ops.neg(partner), partner)
            output = ops.add(ops.mul(value, cos), ops.mul(rotated, sin))
            output = ops.where(rotary_mask, output, value)
            return ops.to_dtype(output, input_dtype, src_dtype=compute_dtype)

        return Pointwise.create(
            device=tensor.get_device(),
            dtype=input_dtype,
            inner_fn=inner_fn,
            ranges=tensor.get_size(),
        )

    @register_lowering(torch.ops.lightx2v.metax_rotary_embedding.default, type_promotion_kind=None)
    def _metax_rotary_embedding_lowering(query, key, positions, freqs, head_size, split_half):
        return (
            rotate(query, positions, freqs, head_size, split_half),
            rotate(key, positions, freqs, head_size, split_half),
        )

    @register_lowering(torch.ops.lightx2v.metax_rotary_embedding_single.default, type_promotion_kind=None)
    def _metax_rotary_embedding_single_lowering(tensor, positions, freqs, head_size, split_half):
        return rotate(tensor, positions, freqs, head_size, split_half)


@PLATFORM_ROPE_REGISTER("metax_rope")
class MetaxRope(RopeTemplate):
    def __init__(self, layout="interleaved", compute_dtype=torch.float32):
        super().__init__(layout=layout, compute_dtype=compute_dtype)
        _load_metax_rotary_embedding_op()
        _register_metax_rotary_embedding_lowering()

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

        query = query.contiguous()
        key = None if key is None else key.contiguous()
        token_count = query.numel() // (query_heads * head_size)
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

        if key is None:
            output = metax_rotary_embedding_single(
                query,
                positions,
                freqs,
                head_size,
                self.layout == "split_half",
            )
            return output

        return metax_rotary_embedding(
            query,
            key,
            positions,
            freqs,
            head_size,
            self.layout == "split_half",
        )

    def apply(self, query, key, freqs, positions=None, rotary_dim=None, **kwargs):
        return self._apply(query, key, freqs, positions, rotary_dim)

    def apply_single(self, tensor, freqs, positions=None, rotary_dim=None, **kwargs):
        return self._apply(tensor, None, freqs, positions, rotary_dim)


__all__ = ["MetaxRope"]
