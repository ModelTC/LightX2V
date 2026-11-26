import torch
import triton
import triton.language as tl
from torch.distributed._symmetric_memory import get_symm_mem_workspace

from .infer import get_self_attention


@triton.jit
def _send_qkv(
    Q, K, V, peers, rows, rank, prefix, Q_STRIDES: tl.constexpr, K_STRIDES: tl.constexpr, V_STRIDES: tl.constexpr, SIZE: tl.constexpr, WIDTH: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr
):
    peer = tl.program_id(0) % SIZE
    index = (tl.program_id(0) // SIZE * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    row, col = index // WIDTH, peer * WIDTH + index % WIDTH
    head, channel = col // DIM, col % DIM
    mask = row < rows
    q = tl.load(Q + row * Q_STRIDES[0] + head * Q_STRIDES[1] + channel * Q_STRIDES[2], mask, other=0)
    k = tl.load(K + row * K_STRIDES[0] + head * K_STRIDES[1] + channel * K_STRIDES[2], mask, other=0)
    v = tl.load(V + row * V_STRIDES[0] + head * V_STRIDES[1] + channel * V_STRIDES[2], mask, other=0)
    dest = tl.load(peers.to(tl.pointer_type(tl.uint64)) + peer).to(tl.pointer_type(Q.dtype.element_ty))
    q_size = rows.to(tl.int64) * SIZE * WIDTH
    kv_rows = rows.to(tl.int64) + prefix
    # Each peer receives one head shard from every rank, in spatial rank order.
    kv_offset = q_size + (rank * kv_rows + prefix) * WIDTH + index
    tl.store(dest + rank * rows.to(tl.int64) * WIDTH + index, q, mask)
    tl.store(dest + kv_offset, k, mask)
    tl.store(dest + kv_offset + SIZE * kv_rows * WIDTH, v, mask)


@triton.jit
def _send_output(X, peers, rows, rank, output_offset, SIZE: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    peer = tl.program_id(0) % SIZE
    index = (tl.program_id(0) // SIZE * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    row = index // WIDTH
    value = tl.load(X + peer * rows.to(tl.int64) * WIDTH + index, row < rows, other=0)
    dest = tl.load(peers.to(tl.pointer_type(tl.uint64)) + peer).to(tl.pointer_type(X.dtype.element_ty))
    tl.store(dest + output_offset + (row * SIZE + rank) * WIDTH + index % WIDTH, value, row < rows)


@torch.library.custom_op("lightx2v::avatar_nvlink_attention", mutates_args=(), device_types="cuda", tags=(torch.Tag.cudagraph_unsafe,))
def nvlink_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cached: list[torch.Tensor],
    frame_tokens: int,
    num_cond_latents: int,
    save_prefix: bool,
    group_name: str,
    size: int,
    attn_type: str,
) -> list[torch.Tensor]:
    rows, heads, dim = q.shape
    shard_heads = heads // size
    width = shard_heads * dim
    prefix = cached[0].size(1) if cached else 0
    kv_size = (rows + prefix) * heads * dim
    workspace = get_symm_mem_workspace(group_name, 4 * kv_size * q.element_size())
    rank = workspace.rank
    # Reserve three KV-sized regions for Q/K/V and one for the returned heads.
    qkv = workspace.get_buffer(rank, (q.numel() + 2 * kv_size,), q.dtype)
    queries, keys, values = (part.view(-1, shard_heads, dim) for part in qkv.split((q.numel(), kv_size, kv_size)))
    if cached:
        keys.view(size, rows + prefix, shard_heads, dim)[:, :prefix].copy_(cached[0])
        values.view(size, rows + prefix, shard_heads, dim)[:, :prefix].copy_(cached[1])
    block = 1024
    grid = (size * triton.cdiv(rows * width, block),)
    _send_qkv[grid](q, k, v, workspace.buffer_ptrs_dev, rows, rank, prefix, q.stride(), k.stride(), v.stride(), size, width, dim, block)
    workspace.barrier()
    new_cache = {} if save_prefix else None
    attended = get_self_attention(attn_type, size).apply(queries, keys, values, frame_tokens, num_cond_latents, new_cache).contiguous()
    # Return heads in a disjoint region. The next QKV exchange can start while peers copy their output.
    output_offset = workspace.buffer_size // q.element_size() - q.numel()
    _send_output[grid](attended, workspace.buffer_ptrs_dev, rows, rank, output_offset, size, width, block)
    workspace.barrier()
    result = workspace.get_buffer(rank, (rows, heads * dim), q.dtype, storage_offset=output_offset).clone()
    return [result, new_cache["k"], new_cache["v"]] if save_prefix else [result]


@nvlink_attention.register_fake
def _(q, k, v, cached, frame_tokens, num_cond_latents, save_prefix, group_name, size, attn_type):
    rows, heads, dim = q.shape
    result = q.new_empty((rows, heads * dim))
    shape = (size, frame_tokens * num_cond_latents, heads // size, dim)
    return [result, k.new_empty(shape), v.new_empty(shape)] if save_prefix else [result]


def nvlink_self_attention(q, k, v, frame_tokens, num_cond_latents, layer_cache, *, group_name, size, attn_type):
    cached = [layer_cache["k"], layer_cache["v"]] if layer_cache is not None and not num_cond_latents else []
    save_prefix = layer_cache is not None and num_cond_latents > 0
    output = nvlink_attention(q, k, v, cached, frame_tokens, num_cond_latents, save_prefix, group_name, size, attn_type)
    if save_prefix:
        layer_cache["k"], layer_cache["v"] = output[1:]
    return output[0]
