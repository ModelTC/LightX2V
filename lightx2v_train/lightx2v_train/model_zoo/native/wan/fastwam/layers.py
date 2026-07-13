import torch
import torch.nn.functional as F


def timestep_embedding(dim: int, position: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.pow(
        10000,
        -torch.arange(half, dtype=torch.float64, device=position.device) / half,
    )
    embedding = torch.outer(position.to(torch.float64), frequencies)
    return torch.cat([torch.cos(embedding), torch.sin(embedding)], dim=1).to(position.dtype)


def apply_rope(x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    complex_x = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    complex_frequencies = frequencies.to(device=x.device, dtype=torch.complex128)
    return torch.view_as_real(complex_x * complex_frequencies).flatten(-2).to(dtype)


def scaled_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    attention_mask = None
    if mask is not None:
        attention_mask = mask.to(device=query.device, dtype=torch.bool)
        if attention_mask.ndim == 2:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        elif attention_mask.ndim == 3:
            attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.ndim != 4:
            raise ValueError(f"Attention mask must be 2D, 3D, or 4D, got {attention_mask.ndim}D")

    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attention_mask,
    )
    return output.transpose(1, 2).contiguous()


def split_modulation(block, time_modulation: torch.Tensor):
    has_sequence_dimension = time_modulation.ndim == 4
    chunk_dimension = 2 if has_sequence_dimension else 1
    values = (block.modulation.to(time_modulation) + time_modulation).chunk(6, dim=chunk_dimension)
    if has_sequence_dimension:
        values = tuple(value.squeeze(2) for value in values)
    return values


def layer_norm(module, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(module, torch.nn.LayerNorm):
        return module(value)
    weight = None if module.weight is None else module.weight.float()
    bias = None if module.bias is None else module.bias.float()
    return F.layer_norm(
        value.float(), module.normalized_shape, weight, bias, module.eps
    ).to(value.dtype)


def masked_cross_attention(block, x, context, context_mask=None):
    batch_size, _, num_heads = x.shape[0], x.shape[1], block.cross_attn.num_heads
    head_dim = block.cross_attn.head_dim
    query = block.cross_attn.norm_q(block.cross_attn.q(x)).view(batch_size, -1, num_heads, head_dim)
    key = block.cross_attn.norm_k(block.cross_attn.k(context)).view(batch_size, -1, num_heads, head_dim)
    value = block.cross_attn.v(context).view(batch_size, -1, num_heads, head_dim)
    output = scaled_attention(query, key, value, context_mask)
    return block.cross_attn.o(output.flatten(2))


def expert_block_forward(
    block,
    x: torch.Tensor,
    context: torch.Tensor,
    time_modulation: torch.Tensor,
    frequencies: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    self_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = split_modulation(
        block, time_modulation
    )
    attention_input = layer_norm(block.norm1, x) * (1 + scale_msa) + shift_msa
    batch_size, sequence_length = attention_input.shape[:2]
    num_heads = block.self_attn.num_heads
    head_dim = block.self_attn.head_dim
    query = block.self_attn.norm_q(block.self_attn.q(attention_input)).view(
        batch_size, sequence_length, num_heads, head_dim
    )
    key = block.self_attn.norm_k(block.self_attn.k(attention_input)).view(
        batch_size, sequence_length, num_heads, head_dim
    )
    value = block.self_attn.v(attention_input).view(
        batch_size, sequence_length, num_heads, head_dim
    )
    attention = scaled_attention(
        apply_rope(query, frequencies),
        apply_rope(key, frequencies),
        value,
        self_attention_mask,
    )
    x = x + gate_msa * block.self_attn.o(attention.flatten(2))
    x = x + masked_cross_attention(block, layer_norm(block.norm3, x), context, context_mask)
    mlp_input = layer_norm(block.norm2, x) * (1 + scale_mlp) + shift_mlp
    return x + gate_mlp * block.ffn(mlp_input)
