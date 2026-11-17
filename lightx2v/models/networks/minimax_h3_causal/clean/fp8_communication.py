"""Packed FP8 Ulysses exchange for causal H3, using the shared Triton kernels."""

from typing import NamedTuple

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _resolve_process_group


def fp8_packet_layout(rows: int, dim: int) -> tuple[int, int]:
    """FP32 scale offset and 512-byte-aligned peer packet size."""
    scale_offset = (rows * dim + 3) // 4
    stride = ((scale_offset + rows) * 4 + 511) // 512 * 512
    return scale_offset, stride


# The FP8 and FP32 views share storage. Keep them inside custom ops so
# torch.compile cannot functionalize them into independent output buffers.
@torch.library.custom_op("lightx2v::h3_causal_exchange_qkv_fp8", mutates_args=(), device_types="cuda")
def exchange_qkv_fp8(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, group_name: str, size: int) -> torch.Tensor:
    import triton

    from lightx2v.common.ops.attn.kernels import ulysses_layout as kernels

    group = _resolve_process_group(group_name)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Packed FP8 communication requires equal Q/K/V shapes")
    length, heads, dim = q.shape
    kernels._check_hidden_dims(dim, "packed QKV fp8 path")
    shard_heads, head_offset, head_stride = kernels._resolve_qkv_head_layout(heads, size, None)
    rows = length * 3 * shard_heads
    # Each peer gets one packet: FP8 payload, aligned FP32 scales, then padding.
    # Align peer packets to 512 bytes for NCCL transfers.
    scale_offset, stride = fp8_packet_layout(rows, dim)
    packed = torch.empty((size, stride), device=q.device, dtype=torch.uint8)
    total_rows = size * length * shard_heads
    kernels._qkv_pre_fp8_kernel[(triton.cdiv(total_rows, 8),)](
        q,
        k,
        v,
        packed.view(torch.float8_e4m3fn),
        packed.view(torch.float32),
        total_rows,
        length,
        shard_heads,
        dim,
        q.stride(),
        k.stride(),
        v.stride(),
        head_offset,
        head_stride,
        stride,
        scale_offset,
        stride // 4,
        8,
        triton.next_power_of_2(dim),
        num_warps=4,
    )
    received = torch.empty_like(packed)
    dist.all_to_all_single(received, packed, group=group)
    return received


@exchange_qkv_fp8.register_fake
def exchange_qkv_fp8_fake(q, k, v, group_name, size):
    rows = q.shape[0] * 3 * (q.shape[1] // size)
    _, stride = fp8_packet_layout(rows, q.shape[2])
    return q.new_empty((size, stride), dtype=torch.uint8)


class FP8QKV(NamedTuple):
    """Received peer packets and the valid, unpadded attention shape."""

    data: torch.Tensor
    shape: tuple[int, int, int]
    dtype: torch.dtype

    def unpack(self):
        """Materialize QKV for RoPE backends outside the fused BF16 path."""
        import triton

        from lightx2v.common.ops.attn.kernels import ulysses_layout as kernels

        length, heads, dim = self.shape
        size = self.data.shape[0]
        local_length = (length + size - 1) // size
        scale_offset, stride = fp8_packet_layout(local_length * 3 * heads, dim)
        shape = (size * local_length, heads, dim)
        outputs = tuple(torch.empty(shape, device=self.data.device, dtype=self.dtype) for _ in range(3))
        total_rows = shape[0] * heads
        kernels._qkv_post_fp8_kernel[(triton.cdiv(total_rows, 16),)](
            self.data.view(torch.float8_e4m3fn),
            self.data.view(torch.float32),
            *outputs,
            *outputs,
            total_rows,
            0,
            0,
            size,
            heads,
            dim,
            heads * size,
            0,
            heads,
            stride,
            scale_offset,
            stride // 4,
            True,
            16,
            triton.next_power_of_2(dim),
            num_warps=4,
        )
        return tuple(value[:length] for value in outputs)


@torch.library.custom_op("lightx2v::h3_causal_exchange_output_fp8", mutates_args=(), device_types="cuda")
def exchange_output_fp8(value: torch.Tensor, group_name: str, size: int) -> torch.Tensor:
    import triton

    from lightx2v.common.ops.attn.kernels import ulysses_layout as kernels

    kernels._check_cuda_contiguous(value)
    length, shard_heads, dim = value.shape
    kernels._check_hidden_dims(dim, "packed attention fp8 path")
    # Match the tail padding applied before sequence sharding, without a copy.
    local_length = (length + size - 1) // size
    rows = shard_heads * local_length
    scale_offset, stride = fp8_packet_layout(rows, dim)
    packed = torch.empty((size, stride), device=value.device, dtype=torch.uint8)
    total_rows = size * rows
    kernels._attn_pre_fp8_kernel[(triton.cdiv(total_rows, 8),)](
        value,
        packed.view(torch.float8_e4m3fn),
        packed.view(torch.float32),
        total_rows,
        local_length,
        length,
        shard_heads,
        dim,
        stride,
        scale_offset,
        stride // 4,
        8,
        triton.next_power_of_2(dim),
        num_warps=4,
    )
    received = torch.empty_like(packed)
    dist.all_to_all_single(received, packed, group=_resolve_process_group(group_name))
    output = torch.empty((local_length, size * shard_heads, dim), device=value.device, dtype=value.dtype)
    kernels._attn_post_fp8_kernel[(triton.cdiv(total_rows, 16),)](
        received.view(torch.float8_e4m3fn),
        received.view(torch.float32),
        output,
        total_rows,
        size,
        shard_heads,
        dim,
        stride,
        scale_offset,
        stride // 4,
        16,
        triton.next_power_of_2(dim),
        num_warps=4,
    )
    return output


@exchange_output_fp8.register_fake
def exchange_output_fp8_fake(value, group_name, size):
    return value.new_empty(((value.shape[0] + size - 1) // size, value.shape[1] * size, value.shape[2]))
