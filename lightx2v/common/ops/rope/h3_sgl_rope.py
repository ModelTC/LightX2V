# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from lightx2v.utils.registry_factory import ROPE_REGISTER

from .template import RopeTemplate

# Adapted from SGLang commit 8ef646a5c65bd2f8922483057dddc02e2b0de18c.


def _require_nvidia_triton(tensor: torch.Tensor) -> None:
    if tensor.device.type == "cuda" and getattr(torch.version, "hip", None) is not None:
        raise RuntimeError("H3 RoPE exact parity kernel supports NVIDIA CUDA only")


@triton.jit
def _round_bf16_to_fp32(value):
    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _round_fp16_to_fp32(value):
    rounded = tl.inline_asm_elementwise(
        asm="cvt.rn.f16.f32 $0, $1;",
        constraints="=h,f",
        args=[value],
        dtype=tl.float16,
        is_pure=True,
        pack=1,
    )
    return rounded.to(tl.float32)


@triton.jit
def _qk_neox_rope_kernel(
    q_ptr,
    k_ptr,
    cache_ptr,
    positions_ptr,
    q_rows,
    q_heads,
    k_heads,
    head_dim,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    position_count,
    ROPE_DIM: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    is_k = pid >= q_rows
    row = pid - q_rows if is_k else pid
    heads = k_heads if is_k else q_heads
    token = row // heads
    head = row % heads
    tensor_ptr = k_ptr if is_k else q_ptr
    token_stride = k_token_stride if is_k else q_token_stride
    head_stride = k_head_stride if is_k else q_head_stride

    half = ROPE_DIM // 2
    offsets = tl.arange(0, BLOCK_HALF)
    mask = offsets < half
    base = token * token_stride + head * head_stride
    position = tl.load(positions_ptr + token % position_count)
    cache_base = position * ROPE_DIM

    first = tl.load(tensor_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    second = tl.load(tensor_ptr + base + half + offsets, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cache_ptr + cache_base + offsets, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(cache_ptr + cache_base + half + offsets, mask=mask, other=0.0).to(tl.float32)

    # Round each product to the activation dtype before the final add/sub.
    # The helpers are optimization barriers, so Triton cannot contract an FMA.
    if IS_BF16:
        first_cos = _round_bf16_to_fp32(first * cos)
        second_sin = _round_bf16_to_fp32(second * sin)
        second_cos = _round_bf16_to_fp32(second * cos)
        first_sin = _round_bf16_to_fp32(first * sin)
    else:
        first_cos = _round_fp16_to_fp32(first * cos)
        second_sin = _round_fp16_to_fp32(second * sin)
        second_cos = _round_fp16_to_fp32(second * cos)
        first_sin = _round_fp16_to_fp32(first * sin)
    out_first = first_cos - second_sin
    out_second = second_cos + first_sin

    tl.store(tensor_ptr + base + offsets, out_first, mask=mask)
    tl.store(tensor_ptr + base + half + offsets, out_second, mask=mask)


def _apply_neox_rope_fallback(
    hidden_states: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    head_dim = hidden_states.shape[-1]
    rotary_dim = cache.shape[-1]
    half = rotary_dim // 2
    flat = hidden_states.reshape(-1, hidden_states.shape[-2], head_dim)
    if flat.shape[0] % positions.numel():
        raise ValueError(f"RoPE position count {positions.numel()} does not divide token count {flat.shape[0]}")
    repeated_positions = positions.repeat(flat.shape[0] // positions.numel())
    selected = cache.index_select(0, repeated_positions)
    cos = selected[:, None, :half]
    sin = selected[:, None, half:]
    first = flat[..., :half]
    second = flat[..., half:rotary_dim]
    first_cos = (first * cos).to(flat.dtype)
    second_sin = (second * sin).to(flat.dtype)
    second_cos = (second * cos).to(flat.dtype)
    first_sin = (first * sin).to(flat.dtype)
    rotated = torch.cat(((first_cos - second_sin).to(flat.dtype), (second_cos + first_sin).to(flat.dtype), flat[..., rotary_dim:]), dim=-1)
    return rotated.reshape(hidden_states.shape)


def _prepare_qk_neox_rope_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim < 2 or k.ndim < 2 or cache.ndim != 2 or positions.ndim != 1:
        raise ValueError(f"H3 RoPE expects Q/K [..., heads, dim], cache [positions, rotary_dim], and positions [tokens]; got {q.shape}, {k.shape}, {cache.shape}, and {positions.shape}")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or cache.dtype != q.dtype:
        raise TypeError(f"H3 RoPE requires matching FP16/BF16 Q/K/cache tensors, got {q.dtype}, {k.dtype}, and {cache.dtype}")
    if positions.dtype is not torch.long:
        raise TypeError(f"H3 RoPE positions must use torch.long, got {positions.dtype}")
    if q.device != k.device or q.device != cache.device or q.device != positions.device:
        raise ValueError("H3 RoPE tensors must be on one device")
    if q.shape[-1] != k.shape[-1] or q.shape[-2] <= 0 or k.shape[-2] <= 0:
        raise ValueError(f"Invalid Q/K shapes for H3 RoPE: {q.shape}, {k.shape}")
    rotary_dim = cache.shape[-1]
    if cache.shape[0] == 0:
        raise ValueError("H3 RoPE cache must contain at least one position")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > q.shape[-1]:
        raise ValueError(f"Invalid rotary dimension {rotary_dim} for head dimension {q.shape[-1]}")
    if positions.numel() == 0:
        raise ValueError("H3 RoPE positions must not be empty")
    q_tokens = q.numel() // (q.shape[-2] * q.shape[-1])
    k_tokens = k.numel() // (k.shape[-2] * k.shape[-1])
    if q_tokens % positions.numel() or k_tokens % positions.numel():
        raise ValueError(f"RoPE position count {positions.numel()} must divide Q/K token counts {q_tokens}/{k_tokens}")
    return cache.contiguous(), positions.contiguous()


def _apply_qk_neox_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache, positions = _prepare_qk_neox_rope_inputs(q, k, cache, positions)
    rotary_dim = cache.shape[-1]
    if q.device.type != "cuda":
        return _apply_neox_rope_fallback(q, cache, positions), _apply_neox_rope_fallback(k, cache, positions)
    _require_nvidia_triton(q)
    q_shape = q.shape
    k_shape = k.shape
    q = q.reshape(-1, q.shape[-2], q.shape[-1])
    k = k.reshape(-1, k.shape[-2], k.shape[-1])

    def safe_row_layout(tensor: torch.Tensor) -> bool:
        heads = tensor.shape[1]
        head_dim = tensor.shape[2]
        head_stride = tensor.stride(1)
        token_span = (heads - 1) * head_stride + head_dim
        return tensor.stride(2) == 1 and (heads <= 1 or head_stride >= head_dim) and (tensor.shape[0] <= 1 or tensor.stride(0) >= token_span)

    if not safe_row_layout(q):
        q = q.contiguous()
    if not safe_row_layout(k):
        k = k.contiguous()
    q_rows = q.shape[0] * q.shape[-2]
    k_rows = k.shape[0] * k.shape[-2]
    if q_rows + k_rows == 0:
        return q.reshape(q_shape), k.reshape(k_shape)
    with torch.cuda.device(q.device):
        _qk_neox_rope_kernel[(q_rows + k_rows,)](
            q,
            k,
            cache,
            positions,
            q_rows,
            q.shape[-2],
            k.shape[-2],
            q.shape[-1],
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            positions.numel(),
            ROPE_DIM=rotary_dim,
            BLOCK_HALF=triton.next_power_of_2(rotary_dim // 2),
            IS_BF16=q.dtype is torch.bfloat16,
            num_warps=1,
        )
    return q.reshape(q_shape), k.reshape(k_shape)


@ROPE_REGISTER("h3_sgl_rope")
class MiniMaxH3SGLRope(RopeTemplate):
    def __init__(self, layout="split_half", compute_dtype=torch.bfloat16):
        if layout != "split_half":
            raise ValueError("MiniMax-H3 SGL RoPE requires split_half layout")
        super().__init__(layout=layout, compute_dtype=compute_dtype)

    def prepare_freqs(self, freqs, rotary_dim: int | None = None):
        if not isinstance(freqs, tuple) or len(freqs) != 2:
            raise TypeError("MiniMax-H3 SGL RoPE expects a (cos, sin) tuple")
        cos, sin = freqs
        if cos.shape != sin.shape or cos.device != sin.device:
            raise ValueError(f"MiniMax-H3 RoPE cos/sin tensors must match, got {cos.shape} and {sin.shape}")
        if cos.ndim == 2:
            if cos.shape[-1] % 2:
                raise ValueError(f"MiniMax-H3 RoPE width must be even, got {cos.shape[-1]}")
            half = cos.shape[-1] // 2
            cache = torch.cat((cos[:, :half], sin[:, :half]), dim=-1)
        elif cos.ndim == 4 and cos.shape[0] == 1 and cos.shape[2] == 1:
            if cos.shape[-1] % 2:
                raise ValueError(f"MiniMax-H3 VAE RoPE width must be even, got {cos.shape[-1]}")
            half = cos.shape[-1] // 2
            cache = torch.cat((cos[0, :, 0, :half], sin[0, :, 0, :half]), dim=-1)
        else:
            raise ValueError(f"Unsupported MiniMax-H3 RoPE frequency shape {cos.shape}")
        cache = cache.to(dtype=self.compute_dtype).contiguous()
        positions = torch.arange(cache.shape[0], device=cache.device, dtype=torch.long)
        return cache, positions

    @staticmethod
    def _is_prepared(freqs) -> bool:
        return isinstance(freqs, tuple) and len(freqs) == 2 and torch.is_tensor(freqs[0]) and torch.is_tensor(freqs[1]) and freqs[0].ndim == 2 and freqs[1].dtype == torch.long

    def apply(self, q: torch.Tensor, k: torch.Tensor, freqs, **kwargs):
        if kwargs.get("materialize", False):
            q, k = q.contiguous(), k.contiguous()
        if not self._is_prepared(freqs):
            freqs = self.prepare_freqs(freqs, rotary_dim=kwargs.get("rotary_dim"))
        cache, positions = freqs
        return _apply_qk_neox_rope(q, k, cache, positions)

    def validate_inputs(self, q: torch.Tensor, k: torch.Tensor, freqs):
        if not self._is_prepared(freqs):
            freqs = self.prepare_freqs(freqs)
        cache, positions = freqs
        return _prepare_qk_neox_rope_inputs(q, k, cache, positions)

    def apply_single(self, x: torch.Tensor, freqs, **kwargs) -> torch.Tensor:
        return self.apply(x, torch.empty_like(x), freqs, **kwargs)[0]


__all__ = ["MiniMaxH3SGLRope"]
