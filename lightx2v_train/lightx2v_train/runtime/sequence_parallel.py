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


def balanced_sequence_lengths(length: int, world_size: int | None = None) -> tuple[int, ...]:
    """Return contiguous, nearly-even sequence lengths for every SP rank.

    Unlike :func:`shrink_sequence`, this layout also supports sequence lengths
    which are not divisible by the SP world size. The first ``remainder``
    ranks own one additional row, so rank-order concatenation reconstructs
    the original sequence.
    """

    length = int(length)
    if length < 0:
        raise ValueError(f"Sequence length must be non-negative, got {length}.")
    if world_size is None:
        world_size = get_sequence_parallel_world_size() if is_sequence_parallel_enabled() else 1
    world_size = int(world_size)
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    quotient, remainder = divmod(length, world_size)
    return tuple(quotient + (rank < remainder) for rank in range(world_size))


def balanced_sequence_slice(length: int, rank: int | None = None, world_size: int | None = None) -> tuple[int, int]:
    """Return the ``[start, end)`` interval owned by an SP rank."""

    if world_size is None:
        world_size = get_sequence_parallel_world_size() if is_sequence_parallel_enabled() else 1
    if rank is None:
        rank = get_sequence_parallel_rank() if is_sequence_parallel_enabled() else 0
    lengths = balanced_sequence_lengths(length, world_size)
    rank = int(rank)
    if not 0 <= rank < len(lengths):
        raise ValueError(f"rank={rank} is outside the SP world size {len(lengths)}.")
    start = sum(lengths[:rank])
    return start, start + lengths[rank]


def shrink_sequence_balanced(tensor: Tensor, dim: int = 1) -> Tensor:
    """Shard a sequence contiguously without requiring exact divisibility."""

    if not is_sequence_parallel_enabled():
        return tensor
    start, end = balanced_sequence_slice(tensor.shape[dim])
    return tensor.narrow(dim, start, end - start).contiguous()


class _AllGatherVariable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, lengths):
        ctx.dim = int(dim)
        ctx.lengths = tuple(int(value) for value in lengths)
        rank = get_sequence_parallel_rank()
        if input_.shape[ctx.dim] != ctx.lengths[rank]:
            raise ValueError(f"Local sequence length {input_.shape[ctx.dim]} does not match lengths[{rank}]={ctx.lengths[rank]}.")

        moved = input_.movedim(ctx.dim, 0).contiguous()
        max_length = max(ctx.lengths, default=0)
        if moved.shape[0] < max_length:
            padding = moved.new_zeros((max_length - moved.shape[0], *moved.shape[1:]))
            moved = torch.cat((moved, padding), dim=0)
        gathered = [torch.empty_like(moved) for _ in ctx.lengths]
        dist.all_gather(gathered, moved, group=get_sequence_parallel_group())
        output = torch.cat(
            [shard.narrow(0, 0, shard_length) for shard, shard_length in zip(gathered, ctx.lengths)],
            dim=0,
        )
        return output.movedim(0, ctx.dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        rank = get_sequence_parallel_rank()
        start = sum(ctx.lengths[:rank])
        grad_input = grad_output.narrow(ctx.dim, start, ctx.lengths[rank])
        return grad_input.contiguous(), None, None


def all_gather_variable_sequence(tensor: Tensor, lengths, dim: int = 1) -> Tensor:
    """Gather unequal contiguous SP shards with local-slice autograd.

    ``lengths`` lists the rows contributed by every rank. As with
    :func:`all_gather_sequence`, backward deliberately returns only this
    rank's slice; parameter gradients are combined once later by
    :func:`sync_sequence_parallel_gradients`.
    """

    if not is_sequence_parallel_enabled():
        return tensor
    lengths = tuple(int(value) for value in lengths)
    if len(lengths) != get_sequence_parallel_world_size():
        raise ValueError(f"Expected {get_sequence_parallel_world_size()} SP shard lengths, got {len(lengths)}.")
    return _AllGatherVariable.apply(tensor, dim, lengths)


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


def _validate_variable_a2a_lengths(sequence_lengths, world_size):
    sequence_lengths = tuple(int(value) for value in sequence_lengths)
    if len(sequence_lengths) != world_size:
        raise ValueError(f"Expected {world_size} sequence shard lengths, got {len(sequence_lengths)}.")
    if any(value < 0 for value in sequence_lengths):
        raise ValueError(f"Sequence shard lengths must be non-negative, got {sequence_lengths}.")
    return sequence_lengths


def _all_to_all_4d_variable(
    input_: Tensor,
    scatter_dim: int,
    gather_dim: int,
    sequence_lengths,
    group,
) -> Tensor:
    """Uneven Ulysses all-to-all over ``[batch, sequence, heads, dim]``.

    The sequence shards may differ by one row. The sequence-to-head direction
    sends one contiguous head shard to every peer; the inverse sends each
    peer's contiguous sequence interval back and joins the received head
    shards. No sequence padding or quadratic attention mask is introduced.
    """

    if input_.dim() != 4:
        raise ValueError(f"all_to_all_4d_variable expects a 4D tensor, got shape={tuple(input_.shape)}.")

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    sequence_lengths = _validate_variable_a2a_lengths(sequence_lengths, world_size)
    batch, sequence, heads, head_dim = input_.shape

    if scatter_dim == 2 and gather_dim == 1:
        if sequence != sequence_lengths[rank]:
            raise ValueError(f"Rank {rank} owns {sequence} rows, but sequence_lengths[{rank}]={sequence_lengths[rank]}.")
        if heads % world_size != 0:
            raise ValueError(f"num_heads={heads} must be divisible by sp_size={world_size}.")
        shard_heads = heads // world_size
        send_chunks = [input_.narrow(2, peer * shard_heads, shard_heads).contiguous().reshape(-1) for peer in range(world_size)]
        input_flat = torch.cat(send_chunks, dim=0)
        input_split_sizes = [batch * sequence * shard_heads * head_dim] * world_size
        output_split_sizes = [batch * shard_length * shard_heads * head_dim for shard_length in sequence_lengths]
        output_flat = input_.new_empty(sum(output_split_sizes))
        dist.all_to_all_single(
            output_flat,
            input_flat,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        received = [chunk.reshape(batch, shard_length, shard_heads, head_dim) for chunk, shard_length in zip(torch.split(output_flat, output_split_sizes), sequence_lengths)]
        return torch.cat(received, dim=1).contiguous()

    if scatter_dim == 1 and gather_dim == 2:
        global_sequence = sum(sequence_lengths)
        if sequence != global_sequence:
            raise ValueError(f"Head-sharded attention tensor has sequence length {sequence}, expected {global_sequence}.")
        send_chunks = [chunk.contiguous().reshape(-1) for chunk in torch.split(input_, sequence_lengths, dim=1)]
        input_flat = torch.cat(send_chunks, dim=0)
        input_split_sizes = [batch * shard_length * heads * head_dim for shard_length in sequence_lengths]
        local_sequence = sequence_lengths[rank]
        output_split_sizes = [batch * local_sequence * heads * head_dim] * world_size
        output_flat = input_.new_empty(sum(output_split_sizes))
        dist.all_to_all_single(
            output_flat,
            input_flat,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        received = [chunk.reshape(batch, local_sequence, heads, head_dim) for chunk in torch.split(output_flat, output_split_sizes)]
        return torch.cat(received, dim=2).contiguous()

    raise ValueError("all_to_all_4d_variable only supports scatter/gather dim pairs (2, 1) and (1, 2).")


class _AllToAll4DVariable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, scatter_dim, gather_dim, sequence_lengths):
        ctx.scatter_dim = int(scatter_dim)
        ctx.gather_dim = int(gather_dim)
        ctx.sequence_lengths = tuple(int(value) for value in sequence_lengths)
        ctx.group = get_sequence_parallel_group()
        return _all_to_all_4d_variable(
            input_,
            ctx.scatter_dim,
            ctx.gather_dim,
            ctx.sequence_lengths,
            ctx.group,
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _all_to_all_4d_variable(
            grad_output,
            ctx.gather_dim,
            ctx.scatter_dim,
            ctx.sequence_lengths,
            ctx.group,
        )
        return grad_input, None, None, None


def all_to_all_4d_variable(
    tensor: Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    *,
    sequence_lengths,
) -> Tensor:
    """Autograd-aware uneven Ulysses all-to-all.

    ``sequence_lengths`` describes the contiguous local row count for every
    SP rank and is shared by the forward and inverse transformations.
    """

    if not is_sequence_parallel_enabled():
        return tensor
    return _AllToAll4DVariable.apply(
        tensor,
        scatter_dim,
        gather_dim,
        tuple(sequence_lengths),
    )


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


@torch.no_grad()
def sync_sequence_parallel_parameters(params, src_sp_rank: int = 0):
    """Make replicated parameters identical inside each SP group.

    This is primarily needed for independently initialized LoRA adapters.
    FSDP2 parameters are DTensors, so only the rank-local DP shard is
    broadcast; SP peers have the same DP coordinate and therefore matching
    local shard layouts.
    """

    if not is_sequence_parallel_enabled():
        return
    group = get_sequence_parallel_group()
    src = get_sequence_parallel_src_rank(src_sp_rank)
    for param in params:
        dist.broadcast(_local_tensor(param.detach()), src=src, group=group)


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
