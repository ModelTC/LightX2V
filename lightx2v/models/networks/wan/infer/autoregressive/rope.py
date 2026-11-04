import torch
import triton

from lightx2v.models.networks.wan.infer.triton_ops import _causal_rope_apply_kernel


def apply_dynamic_causal_rope(q, k, grid_sizes, freqs, first_frame_tensor, complex128_math=False):
    """Apply Wan causal RoPE using a GPU frame offset that can change on replay."""
    if torch.compiler.is_compiling():
        return _dynamic_causal_rope(q, k, grid_sizes, freqs, first_frame_tensor, complex128_math)
    return _apply_dynamic_causal_rope(q, k, grid_sizes, freqs, first_frame_tensor, complex128_math)


@torch.library.custom_op("lightx2v::dynamic_causal_rope", mutates_args=(), device_types="cuda")
def _dynamic_causal_rope(q: torch.Tensor, k: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, first_frame_tensor: torch.Tensor, complex128_math: bool) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply_dynamic_causal_rope(q, k, grid_sizes, freqs, first_frame_tensor, complex128_math)


@_dynamic_causal_rope.register_fake
def _dynamic_causal_rope_fake(q, k, grid_sizes, freqs, first_frame_tensor, complex128_math=False):
    return torch.empty_like(q, memory_format=torch.contiguous_format), torch.empty_like(k, memory_format=torch.contiguous_format)


def apply_dynamic_causal_rope_single(x, grid_sizes, freqs, first_frame_tensor, complex128_math=False):
    if torch.compiler.is_compiling():
        return _dynamic_causal_rope_single(x, grid_sizes, freqs, first_frame_tensor, complex128_math)
    return _apply_dynamic_causal_rope_single(x, grid_sizes, freqs, first_frame_tensor, complex128_math)


@torch.library.custom_op("lightx2v::dynamic_causal_rope_single", mutates_args=(), device_types="cuda")
def _dynamic_causal_rope_single(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, first_frame_tensor: torch.Tensor, complex128_math: bool) -> torch.Tensor:
    return _apply_dynamic_causal_rope_single(x, grid_sizes, freqs, first_frame_tensor, complex128_math)


@_dynamic_causal_rope_single.register_fake
def _dynamic_causal_rope_single_fake(x, grid_sizes, freqs, first_frame_tensor, complex128_math=False):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _apply_dynamic_causal_rope(q, k, grid_sizes, freqs, first_frame_tensor, complex128_math=False):
    grid_sizes = grid_sizes.to(device=q.device, dtype=torch.int64, non_blocking=True).contiguous()
    freqs_real = torch.view_as_real(freqs.to(device=q.device))
    return _rotate_causal_tensor(q, grid_sizes, freqs_real, first_frame_tensor, complex128_math), _rotate_causal_tensor(k, grid_sizes, freqs_real, first_frame_tensor, complex128_math)


def _apply_dynamic_causal_rope_single(x, grid_sizes, freqs, first_frame_tensor, complex128_math=False):
    grid_sizes = grid_sizes.to(device=x.device, dtype=torch.int64, non_blocking=True).contiguous()
    freqs_real = torch.view_as_real(freqs.to(device=x.device))
    return _rotate_causal_tensor(x, grid_sizes, freqs_real, first_frame_tensor, complex128_math)


def _rotate_causal_tensor(tensor, grid_sizes, freqs_real, first_frame_tensor, complex128_math=False):
    sequence_length, num_heads, head_dim = tensor.shape
    half_head_dim = head_dim // 2
    spatial_dim = half_head_dim // 3
    contiguous = tensor.contiguous()
    output = torch.empty_like(contiguous)
    _causal_rope_apply_kernel[(sequence_length * num_heads,)](
        output,
        contiguous,
        freqs_real,
        grid_sizes,
        num_heads,
        head_dim,
        sequence_length,
        half_head_dim - 2 * spatial_dim,
        spatial_dim,
        0,
        False,
        BLOCK_SIZE=triton.next_power_of_2(half_head_dim),
        start_frame_ptr=first_frame_tensor,
        dynamic_start=True,
        complex128_math=complex128_math,
        num_warps=1,
        num_stages=2,
    )
    return output
