from functools import cache

import torch
import torch.distributed as dist


@cache
def _get_round_robin_schedule(world_size, rank):
    """Return the immutable pairwise schedule for one rank."""
    if world_size % 2 != 0:
        raise ValueError(f"round_robin Ulysses A2A requires an even world_size, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}.")

    teams = list(range(world_size))
    schedule = []
    for _ in range(world_size - 1):
        for index in range(world_size // 2):
            left = teams[index]
            right = teams[world_size - 1 - index]
            if rank == left:
                peer = right
                break
            if rank == right:
                peer = left
                break
        schedule.append((peer, rank < peer))
        teams = [teams[0], teams[-1], *teams[1:-1]]
    return tuple(schedule)


class TorchUlyssesA2A:
    """Ulysses all-to-all implemented by ``torch.distributed``."""

    @staticmethod
    def exchange(input_tensor, group=None, async_op=False):
        output_tensor = torch.empty_like(input_tensor)
        work = dist.all_to_all_single(output_tensor, input_tensor, group=group, async_op=async_op)
        return output_tensor, work


class RoundRobinUlyssesA2A:
    """Pairwise round-robin exchange used by the legacy 4090 path."""

    def exchange(self, input_tensor, group=None, async_op=False):
        if async_op:
            raise ValueError("round_robin Ulysses A2A does not support asynchronous exchange.")

        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        if input_tensor.shape[0] != world_size:
            raise ValueError(f"round_robin Ulysses A2A expects dim 0 == world_size ({world_size}), got shape={tuple(input_tensor.shape)}.")

        output_tensor = torch.empty_like(input_tensor)
        output_tensor[rank].copy_(input_tensor[rank])

        for peer, send_first in _get_round_robin_schedule(world_size, rank):
            peer_global_rank = dist.get_global_rank(group, peer) if group is not None else peer
            if send_first:
                send_work = dist.isend(input_tensor[peer], dst=peer_global_rank, group=group)
                recv_work = dist.irecv(output_tensor[peer], src=peer_global_rank, group=group)
                send_work.wait()
                recv_work.wait()
            else:
                recv_work = dist.irecv(output_tensor[peer], src=peer_global_rank, group=group)
                send_work = dist.isend(input_tensor[peer], dst=peer_global_rank, group=group)
                recv_work.wait()
                send_work.wait()

        return output_tensor, None


def create_ulysses_a2a_backend(name):
    if name == "torch":
        return TorchUlyssesA2A()
    if name == "round_robin":
        return RoundRobinUlyssesA2A()
    raise ValueError(f"Unknown a2a_backend={name!r}; expected 'torch' or 'round_robin'.")
