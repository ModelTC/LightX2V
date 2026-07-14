from typing import List

import torch
import torch.distributed as dist
from torch import Tensor

from .common.distributed.advanced import (
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from .common.distributed.basic import get_device
from .common.distributed.ops import Gather
from .common.logger import get_logger
from .common.utils import safe_pad_operation
from .types import MemoryState

logger = get_logger(__name__)


def causal_conv_slice_inputs(x, split_size, memory_state):
    sp_size = get_sequence_parallel_world_size()
    sp_group = get_sequence_parallel_group()
    sp_rank = get_sequence_parallel_rank()
    if sp_group is None:
        return x

    assert memory_state != MemoryState.UNSET
    leave_out = 1 if memory_state != MemoryState.ACTIVE else 0

    # Should have at least sp_size slices.
    num_slices = (x.size(2) - leave_out) // split_size
    assert num_slices >= sp_size, f"{num_slices} < {sp_size}"

    split_sizes = [split_size + leave_out] + [split_size] * (num_slices - 1)
    split_sizes += [x.size(2) - sum(split_sizes)]
    assert sum(split_sizes) == x.size(2)

    split_sizes = torch.tensor(split_sizes)
    slices_per_rank = len(split_sizes) // sp_size
    split_sizes = split_sizes.split([slices_per_rank] * (sp_size - 1) + [len(split_sizes) - slices_per_rank * (sp_size - 1)])
    split_sizes = list(map(lambda s: s.sum().item(), split_sizes))
    logger.debug(f"split_sizes: {split_sizes}")
    return x.split(split_sizes, dim=2)[sp_rank]


def causal_conv_gather_outputs(x):
    sp_group = get_sequence_parallel_group()
    sp_size = get_sequence_parallel_world_size()
    if sp_group is None:
        return x

    # Communicate shapes.
    unpad_lens = torch.empty((sp_size,), device=get_device(), dtype=torch.long)
    local_unpad_len = torch.tensor([x.size(2)], device=get_device(), dtype=torch.long)
    torch.distributed.all_gather_into_tensor(unpad_lens, local_unpad_len, group=sp_group)

    # Padding to max_len for gather.
    max_len = unpad_lens.max()
    x_pad = safe_pad_operation(x, (0, 0, 0, 0, 0, max_len - x.size(2))).contiguous()

    # Gather outputs.
    x_pad = Gather.apply(sp_group, x_pad, 2, True)

    # Remove padding.
    x_pad_lists = list(x_pad.chunk(sp_size, dim=2))
    for i, (x_pad, unpad_len) in enumerate(zip(x_pad_lists, unpad_lens)):
        x_pad_lists[i] = x_pad[:, :, :unpad_len]

    return torch.cat(x_pad_lists, dim=2)


def get_output_len(conv_module, input_len, pad_len, dim=0):
    dilated_kernerl_size = conv_module.dilation[dim] * (conv_module.kernel_size[dim] - 1) + 1
    output_len = (input_len + pad_len - dilated_kernerl_size) // conv_module.stride[dim] + 1
    return output_len


def get_cache_size(conv_module, input_len, pad_len, dim=0):
    dilated_kernerl_size = conv_module.dilation[dim] * (conv_module.kernel_size[dim] - 1) + 1
    output_len = (input_len + pad_len - dilated_kernerl_size) // conv_module.stride[dim] + 1
    remain_len = input_len + pad_len - ((output_len - 1) * conv_module.stride[dim] + dilated_kernerl_size)
    overlap_len = dilated_kernerl_size - conv_module.stride[dim]
    cache_len = overlap_len + remain_len  # >= 0
    logger.debug(f"I:{input_len}, P:{pad_len}, K:{conv_module.kernel_size[dim]}, S:{conv_module.stride[dim]}, O:{output_len}, Cache:{cache_len}")
    assert output_len > 0
    return cache_len


def cache_send_recv(tensor: List[Tensor], cache_size, times, memory=None):
    sp_group = get_sequence_parallel_group()
    sp_rank = get_sequence_parallel_rank()
    sp_size = get_sequence_parallel_world_size()
    send_dst = 0
    recv_src = 0
    recv_buffer = None
    recv_req = None

    logger.debug(f"[sp{sp_rank}] cur_tensors:{[(t.size(), t.dtype) for t in tensor]}, times: {times}")
    if sp_rank == 0 or sp_group is None:
        if memory is not None:
            recv_buffer = memory.to(tensor[0])
        elif times > 0:
            tile_repeat = [1] * tensor[0].ndim
            tile_repeat[2] = times
            recv_buffer = torch.tile(tensor[0][:, :, :1], tile_repeat)

    if cache_size != 0 and sp_group is not None:
        if sp_rank > 0:
            shape = list(tensor[0].size())
            shape[2] = cache_size
            recv_buffer = torch.empty(*shape, device=tensor[0].device, dtype=tensor[0].dtype).contiguous()
            recv_req = dist.irecv(recv_buffer, recv_src, group=sp_group)
        if sp_rank < sp_size - 1:
            if cache_size > tensor[-1].size(2) and len(tensor) == 1:
                logger.debug(f"[sp{sp_rank}] force concat before send {tensor[-1].size()}")
                if recv_req is not None:
                    recv_req.wait()
                tensor[0] = torch.cat([recv_buffer, tensor[0]], dim=2)
                recv_buffer = None
            assert cache_size <= tensor[-1].size(2), f"Not enough value to cache, got {tensor[-1].size()}, cache_size={cache_size}"
            dist.isend(tensor[-1][:, :, -cache_size:].detach().contiguous(), send_dst, group=sp_group)
        if recv_req is not None:
            recv_req.wait()

    logger.debug(f"[sp{sp_rank}] recv_src:{recv_src}, recv_buffer:{recv_buffer.size() if recv_buffer is not None else None}")
    return recv_buffer
