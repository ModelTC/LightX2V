import math

import torch
import torch.nn.functional as F

from lightx2v.common.ops.rope import FlashInferRope, TorchRealRope

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:
    triton = None
    tl = None

try:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
except ImportError:
    apply_rope_with_cos_sin_cache_inplace = None

_COSMOS_TORCH_ROPE = TorchRealRope(layout="split_half")
_COSMOS_FLASH_ROPE = FlashInferRope(layout="split_half")


if triton is not None:

    @triton.jit
    def _split_half_rotary_kernel(
        output_ptr,
        x_ptr,
        cos_ptr,
        sin_ptr,
        num_heads,
        head_size,
        num_tokens,
        stride_x_row,
        stride_cos_row,
        stride_sin_row,
        BLOCK_HS_HALF: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        token_idx = (row_idx // num_heads) % num_tokens

        x_row_ptr = x_ptr + row_idx * stride_x_row
        output_row_ptr = output_ptr + row_idx * stride_x_row
        cos_row_ptr = cos_ptr + token_idx * stride_cos_row
        sin_row_ptr = sin_ptr + token_idx * stride_sin_row

        head_size_half = head_size // 2
        offsets = tl.arange(0, BLOCK_HS_HALF)
        mask = offsets < head_size_half

        x1 = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        x2 = tl.load(x_row_ptr + head_size_half + offsets, mask=mask, other=0.0)
        cos1 = tl.load(cos_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        cos2 = tl.load(cos_row_ptr + head_size_half + offsets, mask=mask, other=0.0).to(tl.float32)
        sin1 = tl.load(sin_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sin2 = tl.load(sin_row_ptr + head_size_half + offsets, mask=mask, other=0.0).to(tl.float32)

        x1_fp32 = x1.to(tl.float32)
        x2_fp32 = x2.to(tl.float32)
        out1 = x1_fp32 * cos1 - x2_fp32 * sin1
        out2 = x2_fp32 * cos2 + x1_fp32 * sin2

        tl.store(output_row_ptr + offsets, out1.to(x1.dtype), mask=mask)
        tl.store(output_row_ptr + head_size_half + offsets, out2.to(x2.dtype), mask=mask)


def apply_split_half_rotary_triton(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    num_tokens, num_heads, head_size = x.shape
    x_reshaped = x.contiguous().view(-1, head_size)
    output_reshaped = output.view(-1, head_size)
    block_hs_half = triton.next_power_of_2(head_size // 2)
    grid = (num_tokens * num_heads,)
    with torch.cuda.device(x.device):
        _split_half_rotary_kernel[grid](
            output_reshaped,
            x_reshaped,
            cos.contiguous(),
            sin.contiguous(),
            num_heads,
            head_size,
            num_tokens,
            x_reshaped.stride(0),
            cos.stride(0),
            sin.stride(0),
            BLOCK_HS_HALF=block_hs_half,
        )
    return output


def apply_cosmos3_rotary(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_type: str = "triton"):
    if query.is_cuda and rope_type in {"flashinfer", "flashinfer_rope"} and apply_rope_with_cos_sin_cache_inplace is not None:
        head_dim = query.shape[-1]
        cache = torch.cat(
            [cos[:, : head_dim // 2].float(), sin[:, : head_dim // 2].float()],
            dim=-1,
        ).contiguous()
        return _COSMOS_FLASH_ROPE.apply(query, key, cache)

    if query.is_cuda and rope_type in {"flashinfer", "flashinfer_rope", "triton"} and triton is not None:
        return apply_split_half_rotary_triton(query, cos, sin), apply_split_half_rotary_triton(key, cos, sin)

    return _COSMOS_TORCH_ROPE.apply(query, key, (cos, sin))


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int = 256,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def get_3d_mrope_ids_text_tokens(num_tokens: int, temporal_offset: int | float, use_float_positions: bool = False):
    if use_float_positions:
        ids = torch.arange(num_tokens, dtype=torch.float32) + temporal_offset
    else:
        ids = torch.arange(num_tokens, dtype=torch.long) + int(temporal_offset)
    return ids.unsqueeze(0).expand(3, -1).contiguous(), temporal_offset + num_tokens


def get_3d_mrope_ids_vae_tokens(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    reset_spatial_indices: bool = True,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
):
    fps_modulation_enabled = fps is not None and grid_t > 1
    effective_base_tcf = base_temporal_compression_factor if base_temporal_compression_factor is not None else temporal_compression_factor
    if fps_modulation_enabled:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        t_index = ((frame_indices + start_frame_offset) / tps * base_tps + temporal_offset).view(-1, 1)
        t_index = t_index.expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten() + int(temporal_offset) + start_frame_offset

    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
    if not reset_spatial_indices:
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset
    if fps_modulation_enabled:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)
    return mrope_ids, math.ceil(mrope_ids.max().item()) + 1


def apply_interleaved_mrope(freqs, rope_axes_dim):
    freqs_t = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):
        length = rope_axes_dim[dim] * 3
        freqs_t[..., offset:length:3] = freqs[dim, ..., offset:length:3]
    return freqs_t


def build_rotary_embeddings(position_ids, head_dim, rope_theta, rope_axes_dim, device, dtype):
    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(1)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()
    with torch.autocast(device_type=position_ids.device.type, enabled=False):
        freqs = inv_freq_expanded @ position_ids_expanded
    freqs = freqs.transpose(2, 3)
    freqs = apply_interleaved_mrope(freqs, rope_axes_dim)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().squeeze(0).to(dtype=dtype), emb.sin().squeeze(0).to(dtype=dtype)
