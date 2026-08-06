import torch

from lightx2v.common.ops.rope import RopeTemplate, TorchRealRope
from lightx2v.utils.registry_factory import ROPE_REGISTER

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:
    triton = None
    tl = None


@ROPE_REGISTER("minimax_h3_triton_rope")
class MiniMaxH3TritonRope(RopeTemplate):
    """Partial split-half RoPE used by MiniMax-H3.

    H3 rotates only the leading RoPE dimensions of each attention head and
    leaves the remaining channels unchanged. CUDA tensors use the local
    Triton kernel; other devices (or environments without Triton) fall back
    to the shared real-valued RoPE implementation with identical layout.
    """

    def __init__(self, layout="split_half", compute_dtype=torch.float32):
        super().__init__(layout=layout, compute_dtype=compute_dtype)
        if layout != "split_half":
            raise ValueError("MiniMaxH3TritonRope only supports split_half layout")
        self.torch_rope = TorchRealRope(layout=layout, compute_dtype=compute_dtype)

    def apply(self, query, key, freqs, rotary_dim=None, **kwargs):
        cos, sin = freqs
        rotary_dim = cos.shape[-1] if rotary_dim is None else rotary_dim
        if query.is_cuda and key.is_cuda and triton is not None:
            return (
                apply_partial_split_half_rotary_triton(query, cos, sin, rotary_dim),
                apply_partial_split_half_rotary_triton(key, cos, sin, rotary_dim),
            )
        return self.torch_rope.apply(query, key, freqs, rotary_dim=rotary_dim, **kwargs)

    def apply_single(self, x, freqs, rotary_dim=None, **kwargs):
        cos, sin = freqs
        rotary_dim = cos.shape[-1] if rotary_dim is None else rotary_dim
        if x.is_cuda and triton is not None:
            return apply_partial_split_half_rotary_triton(x, cos, sin, rotary_dim)
        return self.torch_rope.apply_single(x, freqs, rotary_dim=rotary_dim, **kwargs)


if triton is not None:

    @triton.jit
    def _partial_split_half_rotary_kernel(
        output_ptr,
        x_ptr,
        cos_ptr,
        sin_ptr,
        num_heads,
        num_tokens,
        stride_x_row,
        stride_cos_row,
        stride_sin_row,
        HEAD_SIZE: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        token_idx = (row_idx // num_heads) % num_tokens

        offsets = tl.arange(0, BLOCK_SIZE)
        head_mask = offsets < HEAD_SIZE
        rotary_mask = offsets < ROTARY_DIM
        rotary_half = ROTARY_DIM // 2

        x_row_ptr = x_ptr + row_idx * stride_x_row
        output_row_ptr = output_ptr + row_idx * stride_x_row
        cos_row_ptr = cos_ptr + token_idx * stride_cos_row
        sin_row_ptr = sin_ptr + token_idx * stride_sin_row

        x = tl.load(x_row_ptr + offsets, mask=head_mask, other=0.0)
        partner_offsets = tl.where(offsets < rotary_half, offsets + rotary_half, offsets - rotary_half)
        partner = tl.load(x_row_ptr + partner_offsets, mask=rotary_mask, other=0.0)
        rotated = tl.where(offsets < rotary_half, -partner, partner)
        cos = tl.load(cos_row_ptr + offsets, mask=rotary_mask, other=1.0)
        sin = tl.load(sin_row_ptr + offsets, mask=rotary_mask, other=0.0)

        x_fp32 = x.to(tl.float32)
        rotated_fp32 = rotated.to(tl.float32)
        rotated_output = x_fp32 * cos.to(tl.float32) + rotated_fp32 * sin.to(tl.float32)
        output = tl.where(rotary_mask, rotated_output, x_fp32)
        tl.store(output_row_ptr + offsets, output.to(x.dtype), mask=head_mask)


def apply_partial_split_half_rotary_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for MiniMax-H3 Triton RoPE")
    if not x.is_cuda:
        raise ValueError("MiniMax-H3 Triton RoPE requires a CUDA tensor")
    if x.ndim != 3:
        raise ValueError(f"MiniMax-H3 Triton RoPE expects [L, H, D], got {tuple(x.shape)}")
    if cos.shape != sin.shape or cos.ndim != 2:
        raise ValueError(f"cos and sin must have matching [L, R] shapes, got {tuple(cos.shape)} and {tuple(sin.shape)}")

    num_tokens, num_heads, head_size = x.shape
    rotary_dim = cos.shape[-1] if rotary_dim is None else int(rotary_dim)
    if cos.shape[0] != num_tokens:
        raise ValueError(f"RoPE token count ({cos.shape[0]}) does not match input ({num_tokens})")
    if rotary_dim != cos.shape[-1]:
        raise ValueError(f"rotary_dim ({rotary_dim}) must match the H3 cos/sin width ({cos.shape[-1]})")
    if rotary_dim <= 0 or rotary_dim > head_size or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be positive, even, and <= head_size; got rotary_dim={rotary_dim}, head_size={head_size}")

    x = x.contiguous()
    cos = cos.to(device=x.device).contiguous()
    sin = sin.to(device=x.device).contiguous()
    output = torch.empty_like(x)
    block_size = triton.next_power_of_2(head_size)
    grid = (num_tokens * num_heads,)
    with torch.cuda.device(x.device):
        _partial_split_half_rotary_kernel[grid](
            output,
            x,
            cos,
            sin,
            num_heads,
            num_tokens,
            x.stride(1),
            cos.stride(0),
            sin.stride(0),
            HEAD_SIZE=head_size,
            ROTARY_DIM=rotary_dim,
            BLOCK_SIZE=block_size,
        )
    return output
