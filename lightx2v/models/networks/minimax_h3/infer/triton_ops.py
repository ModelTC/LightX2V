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
    leaves the remaining channels unchanged. CUDA and MUSA tensors use the
    local Triton kernel; other devices (or environments without Triton) fall
    back to the shared real-valued RoPE implementation with identical layout.
    """

    def __init__(self, layout="split_half", compute_dtype=torch.float32):
        super().__init__(layout=layout, compute_dtype=compute_dtype)
        if layout != "split_half":
            raise ValueError("MiniMaxH3TritonRope only supports split_half layout")
        self.torch_rope = TorchRealRope(layout=layout, compute_dtype=compute_dtype)

    def apply(self, query, key, freqs, rotary_dim=None, **kwargs):
        cos, sin = freqs
        rotary_dim = cos.shape[-1] if rotary_dim is None else rotary_dim
        if query.device.type in {"cuda", "musa"} and key.device == query.device and triton is not None:
            return apply_partial_split_half_qk_rotary_triton(query, key, cos, sin, rotary_dim)
        return self.torch_rope.apply(query, key, freqs, rotary_dim=rotary_dim, **kwargs)

    def apply_single(self, x, freqs, rotary_dim=None, **kwargs):
        cos, sin = freqs
        rotary_dim = cos.shape[-1] if rotary_dim is None else rotary_dim
        if x.device.type in {"cuda", "musa"} and triton is not None:
            return apply_partial_split_half_rotary_triton(x, cos, sin, rotary_dim)
        return self.torch_rope.apply_single(x, freqs, rotary_dim=rotary_dim, **kwargs)


@ROPE_REGISTER("minimax_h3_musa_rope")
class MiniMaxH3MusaRope(MiniMaxH3TritonRope):
    def apply(self, query, key, freqs, rotary_dim=None, **kwargs):
        cos, sin = freqs
        rotary_dim = cos.shape[-1] if rotary_dim is None else rotary_dim
        if query.device.type == "musa" and rotary_dim == 96:
            from lightx2v_platform.ops.attn.mthreads_musa.h3_fused_ops import apply_h3_qk_rope_fp32

            return apply_h3_qk_rope_fp32(query, key, cos, sin)
        return super().apply(query, key, freqs, rotary_dim=rotary_dim, **kwargs)


@ROPE_REGISTER("minimax_h3_musa_rope_bf16")
class MiniMaxH3MusaBf16Rope(MiniMaxH3TritonRope):
    def apply(self, query, key, freqs, rotary_dim=None, **kwargs):
        cos, sin = freqs
        rotary_dim = cos.shape[-1] if rotary_dim is None else rotary_dim
        if query.device.type == "musa" and rotary_dim == 96:
            from lightx2v_platform.ops.attn.mthreads_musa.h3_fused_ops import apply_h3_qk_rope

            return apply_h3_qk_rope(query, key, cos, sin)
        return super().apply(query, key, freqs, rotary_dim=rotary_dim, **kwargs)


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
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        row_idx = token_idx * num_heads + head_idx

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

    @triton.jit
    def _partial_split_half_qk_rotary_kernel(
        query_output_ptr,
        key_output_ptr,
        query_ptr,
        key_ptr,
        cos_ptr,
        sin_ptr,
        num_heads,
        stride_x_row,
        stride_cos_row,
        stride_sin_row,
        HEAD_SIZE: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        row_idx = token_idx * num_heads + head_idx

        offsets = tl.arange(0, BLOCK_SIZE)
        head_mask = offsets < HEAD_SIZE
        rotary_mask = offsets < ROTARY_DIM
        rotary_half = ROTARY_DIM // 2
        partner_offsets = tl.where(offsets < rotary_half, offsets + rotary_half, offsets - rotary_half)

        query_row_ptr = query_ptr + row_idx * stride_x_row
        key_row_ptr = key_ptr + row_idx * stride_x_row
        query_output_row_ptr = query_output_ptr + row_idx * stride_x_row
        key_output_row_ptr = key_output_ptr + row_idx * stride_x_row
        cos_row_ptr = cos_ptr + token_idx * stride_cos_row
        sin_row_ptr = sin_ptr + token_idx * stride_sin_row

        cos = tl.load(cos_row_ptr + offsets, mask=rotary_mask, other=1.0)
        sin = tl.load(sin_row_ptr + offsets, mask=rotary_mask, other=0.0)

        query = tl.load(query_row_ptr + offsets, mask=head_mask, other=0.0)
        query_partner = tl.load(query_row_ptr + partner_offsets, mask=rotary_mask, other=0.0)
        query_cos = (query.to(tl.float32) * cos.to(tl.float32)).to(tl.bfloat16)
        query_partner_sin = (query_partner.to(tl.float32) * sin.to(tl.float32)).to(tl.bfloat16)
        query_partner_sin = tl.where(offsets < rotary_half, -query_partner_sin, query_partner_sin)
        query_rotated = (query_cos.to(tl.float32) + query_partner_sin.to(tl.float32)).to(tl.bfloat16)
        query_result = tl.where(rotary_mask, query_rotated, query)
        tl.store(query_output_row_ptr + offsets, query_result, mask=head_mask)

        key = tl.load(key_row_ptr + offsets, mask=head_mask, other=0.0)
        key_partner = tl.load(key_row_ptr + partner_offsets, mask=rotary_mask, other=0.0)
        key_cos = (key.to(tl.float32) * cos.to(tl.float32)).to(tl.bfloat16)
        key_partner_sin = (key_partner.to(tl.float32) * sin.to(tl.float32)).to(tl.bfloat16)
        key_partner_sin = tl.where(offsets < rotary_half, -key_partner_sin, key_partner_sin)
        key_rotated = (key_cos.to(tl.float32) + key_partner_sin.to(tl.float32)).to(tl.bfloat16)
        key_result = tl.where(rotary_mask, key_rotated, key)
        tl.store(key_output_row_ptr + offsets, key_result, mask=head_mask)

    @triton.jit
    def _h3_scale_shift_kernel(
        output,
        x,
        shift,
        scale,
        indices,
        shift_row_stride,
        scale_row_stride,
        n_elements,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        rows = offsets // HIDDEN_SIZE
        columns = offsets - rows * HIDDEN_SIZE
        parameter_rows = tl.load(indices + rows, mask=mask, other=0)

        values = tl.load(x + offsets, mask=mask)
        shifts = tl.load(
            shift + parameter_rows * shift_row_stride + columns,
            mask=mask,
        )
        scales = tl.load(
            scale + parameter_rows * scale_row_stride + columns,
            mask=mask,
        )
        factors = (1.0 + scales).to(tl.bfloat16)
        scaled = (values * factors).to(tl.bfloat16)
        tl.store(output + offsets, (scaled + shifts).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _h3_gate_kernel(
        output,
        residual,
        gate,
        branch,
        indices,
        gate_row_stride,
        n_elements,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        rows = offsets // HIDDEN_SIZE
        columns = offsets - rows * HIDDEN_SIZE
        parameter_rows = tl.load(indices + rows, mask=mask, other=0)

        residual_values = tl.load(residual + offsets, mask=mask)
        gate_values = tl.load(
            gate + parameter_rows * gate_row_stride + columns,
            mask=mask,
        )
        branch_values = tl.load(branch + offsets, mask=mask)
        product = (gate_values * branch_values).to(tl.bfloat16)
        tl.store(output + offsets, (residual_values + product).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _h3_swiglu_kernel(
        output,
        projected,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = columns < HIDDEN_SIZE
        row_offset = row * 2 * HIDDEN_SIZE

        value = tl.load(projected + row_offset + columns, mask=mask)
        gate = tl.load(projected + row_offset + HIDDEN_SIZE + columns, mask=mask)
        sigmoid = (1.0 / (1.0 + tl.exp2(-gate.to(tl.float32) * 1.4426950408889634))).to(tl.bfloat16)
        activated = (gate.to(tl.float32) * sigmoid.to(tl.float32)).to(tl.bfloat16)
        result = (activated.to(tl.float32) * value.to(tl.float32)).to(tl.bfloat16)
        tl.store(output + row * HIDDEN_SIZE + columns, result, mask=mask)


def apply_partial_split_half_rotary_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for MiniMax-H3 Triton RoPE")
    if x.device.type not in {"cuda", "musa"}:
        raise ValueError("MiniMax-H3 Triton RoPE requires a CUDA or MUSA tensor")
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
    grid = (num_tokens, num_heads)
    device_module = torch.cuda if x.device.type == "cuda" else torch.musa
    with device_module.device(x.device):
        torch.library.wrap_triton(_partial_split_half_rotary_kernel)[grid](
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


def apply_partial_split_half_qk_rotary_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.shape != key.shape or query.device != key.device or query.dtype != key.dtype:
        raise ValueError("query and key must have the same shape, device, and dtype")
    if query.ndim != 3:
        raise ValueError(f"MiniMax-H3 Triton RoPE expects [L, H, D], got {tuple(query.shape)}")
    if cos.shape != sin.shape or cos.ndim != 2:
        raise ValueError(f"cos and sin must have matching [L, R] shapes, got {tuple(cos.shape)} and {tuple(sin.shape)}")

    num_tokens, num_heads, head_size = query.shape
    rotary_dim = cos.shape[-1] if rotary_dim is None else int(rotary_dim)
    if cos.shape[0] != num_tokens or rotary_dim != cos.shape[-1]:
        raise ValueError("RoPE frequencies do not match the query/key shape")
    if rotary_dim <= 0 or rotary_dim > head_size or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be positive, even, and <= head_size; got rotary_dim={rotary_dim}, head_size={head_size}")

    query = query.contiguous()
    key = key.contiguous()
    cos = cos.to(device=query.device).contiguous()
    sin = sin.to(device=query.device).contiguous()
    query_output = torch.empty_like(query)
    key_output = torch.empty_like(key)
    grid = (num_tokens, num_heads)
    block_size = triton.next_power_of_2(head_size)
    device_module = torch.cuda if query.device.type == "cuda" else torch.musa
    with device_module.device(query.device):
        torch.library.wrap_triton(_partial_split_half_qk_rotary_kernel)[grid](
            query_output,
            key_output,
            query,
            key,
            cos,
            sin,
            num_heads,
            query.stride(1),
            cos.stride(0),
            sin.stride(0),
            HEAD_SIZE=head_size,
            ROTARY_DIM=rotary_dim,
            BLOCK_SIZE=block_size,
        )
    return query_output, key_output


def apply_h3_scale_shift_triton(x, shift, scale, indices):
    output = torch.empty_like(x)
    block_size = 1024
    grid = (triton.cdiv(x.numel(), block_size),)
    torch.library.wrap_triton(_h3_scale_shift_kernel)[grid](
        output,
        x,
        shift,
        scale,
        indices,
        shift.stride(0),
        scale.stride(0),
        x.numel(),
        HIDDEN_SIZE=x.shape[-1],
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output


def apply_h3_gate_triton(residual, gate, branch, indices):
    output = torch.empty_like(residual)
    block_size = 1024
    grid = (triton.cdiv(residual.numel(), block_size),)
    torch.library.wrap_triton(_h3_gate_kernel)[grid](
        output,
        residual,
        gate,
        branch,
        indices,
        gate.stride(0),
        residual.numel(),
        HIDDEN_SIZE=residual.shape[-1],
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output


def apply_h3_swiglu_triton(projected):
    hidden_size = projected.shape[-1] // 2
    projected = projected.view(-1, 2 * hidden_size)
    output = torch.empty((projected.shape[0], hidden_size), dtype=projected.dtype, device=projected.device)
    block_size = 1024
    grid = (projected.shape[0], triton.cdiv(hidden_size, block_size))
    torch.library.wrap_triton(_h3_swiglu_kernel)[grid](
        output,
        projected,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output
