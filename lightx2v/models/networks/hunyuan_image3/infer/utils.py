import math

import torch
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _linear_tensor_attr(linear, attr):
    tensor = getattr(linear, attr, None)
    if tensor is not None:
        return tensor
    return getattr(linear, f"pin_{attr}", None)


def match_linear_dtype(input_tensor, linear):
    weight = _linear_tensor_attr(linear, "weight")
    bias = _linear_tensor_attr(linear, "bias")
    ref = weight if weight is not None else bias
    if ref is None:
        return input_tensor
    return input_tensor.to(device=ref.device, dtype=ref.dtype)


def apply_linear(linear, input_tensor):
    return linear.apply(match_linear_dtype(input_tensor, linear))


def apply_timestep_embedder(weights, t):
    emb = timestep_embedding(t.reshape(-1), 256)
    emb = apply_linear(weights.linear_1, emb)
    emb = F.gelu(emb)
    return apply_linear(weights.linear_2, emb)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
    cos = cos.to(device=q.device, dtype=q.dtype)
    sin = sin.to(device=q.device, dtype=q.dtype)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def weight_device(module):
    for attr in ("weight", "pin_weight", "bias", "pin_bias"):
        tensor = getattr(module, attr, None)
        if tensor is not None:
            return tensor.device
    return None


def first_weight_device(module):
    device = weight_device(module)
    if device is not None:
        return device
    for child in getattr(module, "_modules", {}).values():
        device = first_weight_device(child)
        if device is not None:
            return device
    return None


def to_device(value, device):
    if value is None or device is None:
        return value
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def apply_mlp(gate_and_up_proj, down_proj, hidden_states, hidden_act="silu"):
    original_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, original_shape[-1])
    projected = apply_linear(gate_and_up_proj, flat)
    if hidden_act == "silu":
        x1, x2 = projected.chunk(2, dim=-1)
        hidden = x1 * F.silu(x2)
    elif hidden_act == "gelu":
        hidden = F.gelu(projected)
    else:
        raise NotImplementedError(f"Unsupported HunyuanImage3 hidden_act: {hidden_act}")
    out = apply_linear(down_proj, hidden)
    return out.reshape(*original_shape)
