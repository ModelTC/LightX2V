# Code source: https://github.com/RiseAI-Sys/ParaVAE/blob/main/paravae/dist/split_gather.py

import torch
import torch.distributed as dist

from lightx2v.models.video_encoders.hf.wan.dist.distributed_env import DistributedEnv

def _gather(patch_hidden_state, dim=-1, group=None):
    group_world_size = DistributedEnv.get_group_world_size()
    local_rank = DistributedEnv.get_local_rank()

    patch_height_list = [torch.empty([1], dtype=torch.int64, device=f"cuda:{local_rank}") for _ in range(group_world_size)]

    dist.all_gather(
        patch_height_list,
        torch.tensor(
            [patch_hidden_state.shape[3]],
            dtype=torch.int64,
            device=f"cuda:{local_rank}"
        ),
        group=DistributedEnv.get_vae_group()
    )

    patch_hidden_state_list = [
        torch.zeros(
            [patch_hidden_state.shape[0], patch_hidden_state.shape[1], patch_hidden_state.shape[2], patch_height_list[i].item(),patch_hidden_state.shape[4]],
            dtype=patch_hidden_state.dtype,
            device=f"cuda:{local_rank}",
            requires_grad=patch_hidden_state.requires_grad
        ) for i in range(group_world_size)
    ]

    dist.all_gather(
        patch_hidden_state_list,
        patch_hidden_state.contiguous(),
        group=DistributedEnv.get_vae_group()
    )
    output = torch.cat(patch_hidden_state_list, dim=3)

    return output

def _split(inputs, dim=-1, group=None):
    group_world_size = DistributedEnv.get_group_world_size()
    rank_in_vae_group = DistributedEnv.get_rank_in_vae_group()
    height = inputs.shape[3]

    start_idx = (height + group_world_size - 1) // group_world_size * rank_in_vae_group
    end_idx = min((height + group_world_size - 1) // group_world_size * (rank_in_vae_group + 1), height)

    return inputs[:, :, :, start_idx: end_idx, :].clone()

class _SplitForwardGatherBackward(torch.autograd.Function):
    """Split the input.

    Args:
        inputs: input matrix.
        dim: dimension
        group: process group
    """

    @staticmethod
    def forward(ctx, inputs, dim, group):
        ctx.group = group
        ctx.dim = dim
        return _split(inputs, dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.dim, ctx.group), None, None


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        inputs: input matrix.
        dim: dimension
        group: process group
    """

    @staticmethod
    def forward(ctx, inputs, dim, group):
        ctx.group = group
        ctx.dim = dim
        return _gather(inputs, dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.dim, ctx.group), None, None


def split_forward_gather_backward(group, inputs, dim):
    return _SplitForwardGatherBackward.apply(inputs, dim, group)

def gather_forward_split_backward(group, inputs, dim):
    return _GatherForwardSplitBackward.apply(inputs, dim, group)
