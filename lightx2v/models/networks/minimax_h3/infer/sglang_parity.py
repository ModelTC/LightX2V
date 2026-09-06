import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.ops.mm.mm_weight import unwrap_tp_weight


def _linear_weight(module) -> torch.Tensor:
    concrete = unwrap_tp_weight(module)
    if concrete.has_lora_branch or concrete.has_diff:
        raise NotImplementedError("MiniMax-H3 merged-QKV parity does not support LoRA or diff weights")
    if concrete.bias is not None:
        raise NotImplementedError("MiniMax-H3 merged-QKV parity expects bias-free Q/K/V projections")
    weight = concrete.weight
    if weight is None:
        raise RuntimeError("MiniMax-H3 merged-QKV parity requires resident Q/K/V weights")
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"MiniMax-H3 merged-QKV parity requires BF16 weights, got {weight.dtype}")
    return weight


def tp_all_gather_last_dim(tensor, group, world_size):
    if world_size == 1:
        return tensor
    tensor = tensor.contiguous()
    input_shape = list(tensor.shape)
    gathered_shape = input_shape.copy()
    gathered_shape[0] *= world_size
    gathered = torch.empty(gathered_shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_into_tensor(gathered, tensor, group=group)
    gathered = gathered.reshape([world_size] + input_shape)
    gathered = gathered.movedim(0, tensor.dim() - 1)
    output_shape = input_shape.copy()
    output_shape[-1] *= world_size
    return gathered.reshape(output_shape)


def project_merged_qkv(weights, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_cuda:
        raise TypeError(f"MiniMax-H3 merged-QKV parity requires a CUDA BF16 activation, got device={hidden_states.device}, dtype={hidden_states.dtype}")

    cache = getattr(weights, "_sglang_parity_qkv_cache", None)
    if cache is None:
        modules = (weights.to_q, weights.to_k, weights.to_v)
        source_weights = tuple(_linear_weight(module) for module in modules)
        # SGLang stores contiguous [out, in] rows ordered Q, K, V per rank.
        fused_weight = torch.cat([weight.t() for weight in source_weights], dim=0).contiguous()
        local_inner_dim = source_weights[0].shape[1]
        cache = (fused_weight, local_inner_dim)
        weights._sglang_parity_qkv_cache = cache
        # The merged matrix replaces three resident weights in parity mode.
        for module in modules:
            unwrap_tp_weight(module).weight = None

    fused_weight, local_inner_dim = cache
    qkv = F.linear(hidden_states, fused_weight)
    return qkv.split(local_inner_dim, dim=-1)


def build_sglang_rope_cache(
    freqs: tuple[torch.Tensor, torch.Tensor],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs
    if cos.shape != sin.shape or cos.ndim != 2 or cos.shape[-1] % 2:
        raise ValueError(f"Expected matching even-width [tokens, rotary_dim] cos/sin, got {cos.shape}, {sin.shape}")
    half = cos.shape[-1] // 2
    cos_sin_cache = torch.cat((cos[:, :half], sin[:, :half]), dim=-1).to(dtype=dtype).contiguous()
    positions = torch.arange(cos.shape[0], device=cos.device, dtype=torch.long)
    return cos_sin_cache, positions
