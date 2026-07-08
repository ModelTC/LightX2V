import torch
import torch.distributed as dist
from torch import Tensor

from lightx2v_train.runtime.distributed import (
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
    get_sequence_parallel_world_size,
    is_sequence_parallel_enabled,
)


def shrink_sequence(tensor: Tensor, dim: int = 1) -> Tensor:
    if not is_sequence_parallel_enabled():
        return tensor
    sp_size = get_sequence_parallel_world_size()
    length = tensor.shape[dim]
    if length % sp_size != 0:
        raise ValueError(f"Cannot sequence-shard dim={dim} length={length} by sp_size={sp_size}.")
    local_length = length // sp_size
    start = get_sequence_parallel_rank() * local_length
    return tensor.narrow(dim, start, local_length).contiguous()


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        ctx.input_size = input_.shape[dim]
        world_size = get_sequence_parallel_world_size()
        group = get_sequence_parallel_group()
        input_ = input_.contiguous()
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_, group=group)
        return torch.cat(tensor_list, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        rank = get_sequence_parallel_rank()
        grad_input = torch.split(grad_output, ctx.input_size, dim=ctx.dim)[rank]
        return grad_input.contiguous(), None


def all_gather_sequence(tensor: Tensor, dim: int = 1) -> Tensor:
    if not is_sequence_parallel_enabled():
        return tensor
    return _AllGather.apply(tensor, dim)


def _all_to_all_4d(input_: Tensor, scatter_dim: int, gather_dim: int, group) -> Tensor:
    if input_.dim() != 4:
        raise ValueError(f"all_to_all_4d expects a 4D tensor, got shape={tuple(input_.shape)}.")

    world_size = dist.get_world_size(group)
    if scatter_dim == 2 and gather_dim == 1:
        batch, shard_seq_len, heads, head_dim = input_.shape
        if heads % world_size != 0:
            raise ValueError(f"num_heads={heads} must be divisible by sp_size={world_size}.")
        shard_heads = heads // world_size
        seq_len = shard_seq_len * world_size
        input_t = input_.reshape(batch, shard_seq_len, world_size, shard_heads, head_dim).transpose(0, 2).contiguous()
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)
        return output.reshape(seq_len, batch, shard_heads, head_dim).transpose(0, 1).contiguous().reshape(batch, seq_len, shard_heads, head_dim)

    if scatter_dim == 1 and gather_dim == 2:
        batch, seq_len, shard_heads, head_dim = input_.shape
        if seq_len % world_size != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by sp_size={world_size}.")
        shard_seq_len = seq_len // world_size
        heads = shard_heads * world_size
        input_t = input_.reshape(batch, world_size, shard_seq_len, shard_heads, head_dim).transpose(0, 3).transpose(0, 1).contiguous().reshape(world_size, shard_heads, shard_seq_len, batch, head_dim)
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)
        return output.reshape(heads, shard_seq_len, batch, head_dim).transpose(0, 2).contiguous().reshape(batch, shard_seq_len, heads, head_dim)

    raise ValueError("all_to_all_4d only supports scatter/gather dim pairs (2, 1) and (1, 2).")


class _AllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, scatter_dim, gather_dim):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = get_sequence_parallel_group()
        return _all_to_all_4d(input_, scatter_dim, gather_dim, ctx.group)

    @staticmethod
    def backward(ctx, grad_output):
        return _AllToAll4D.apply(grad_output, ctx.gather_dim, ctx.scatter_dim), None, None


def all_to_all_4d(tensor: Tensor, scatter_dim: int = 2, gather_dim: int = 1) -> Tensor:
    if not is_sequence_parallel_enabled():
        return tensor
    return _AllToAll4D.apply(tensor, scatter_dim, gather_dim)


def _local_tensor(tensor: Tensor) -> Tensor:
    if hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor


def sync_sequence_parallel_gradients(params):
    if not is_sequence_parallel_enabled():
        return

    group = get_sequence_parallel_group()
    for param in params:
        if param.grad is not None:
            dist.all_reduce(_local_tensor(param.grad), op=dist.ReduceOp.SUM, group=group)


def sequence_parallel_frame_slice(num_frames: int, num_frame_per_chunk: int = 1):
    if not is_sequence_parallel_enabled():
        return 0, int(num_frames), int(num_frames)

    sp_size = get_sequence_parallel_world_size()
    num_frames = int(num_frames)
    num_frame_per_chunk = int(num_frame_per_chunk)
    if num_frames % sp_size != 0:
        raise ValueError(f"num_frames={num_frames} must be divisible by sp_size={sp_size}.")

    local_frames = num_frames // sp_size
    if num_frame_per_chunk > 1 and local_frames % num_frame_per_chunk != 0:
        raise ValueError(f"local_frames={local_frames} must be divisible by num_frame_per_chunk={num_frame_per_chunk} for balanced sequence-parallel teacher forcing.")

    start = get_sequence_parallel_rank() * local_frames
    end = start + local_frames
    return start, end, local_frames


def broadcast_sequence_parallel_tensor(tensor: Tensor, src_sp_rank: int = 0) -> Tensor:
    if not is_sequence_parallel_enabled():
        return tensor
    dist.broadcast(
        tensor,
        src=get_sequence_parallel_src_rank(src_sp_rank),
        group=get_sequence_parallel_group(),
    )
    return tensor


def broadcast_sequence_parallel_object(value, src_sp_rank: int = 0):
    if not is_sequence_parallel_enabled():
        return value
    objects = [value if get_sequence_parallel_rank() == src_sp_rank else None]
    dist.broadcast_object_list(
        objects,
        src=get_sequence_parallel_src_rank(src_sp_rank),
        group=get_sequence_parallel_group(),
    )
    return objects[0]


def broadcast_sequence_parallel_value(value, src_sp_rank: int = 0):
    if not is_sequence_parallel_enabled():
        return value
    if torch.is_tensor(value):
        return broadcast_sequence_parallel_tensor(value, src_sp_rank=src_sp_rank)
    if isinstance(value, dict):
        return {key: broadcast_sequence_parallel_value(item, src_sp_rank=src_sp_rank) for key, item in value.items()}
    if isinstance(value, list):
        return [broadcast_sequence_parallel_value(item, src_sp_rank=src_sp_rank) for item in value]
    if isinstance(value, tuple):
        return tuple(broadcast_sequence_parallel_value(item, src_sp_rank=src_sp_rank) for item in value)
    return broadcast_sequence_parallel_object(value, src_sp_rank=src_sp_rank)
