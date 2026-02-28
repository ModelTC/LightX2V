import math
from contextlib import contextmanager
from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.normalization import RMSNorm
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Conv3d

from .common.distributed.advanced import (
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from .common.logger import get_logger
from .common.utils import safe_pad_operation
from .context_parallel_lib import (
    cache_send_recv,
    get_cache_size,
)
from .global_config import get_norm_limit
from .types import (
    MemoryState,
    _inflation_mode_t,
    _memory_device_t,
)

logger = get_logger(__name__)


@contextmanager
def ignore_padding(model):
    orig_padding = model.padding
    model.padding = (0, 0, 0)
    try:
        yield
    finally:
        model.padding = orig_padding


class InflatedCausalConv3d(Conv3d):
    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t,
        memory_device: _memory_device_t = "same",
        **kwargs,
    ):
        self.inflation_mode = inflation_mode
        self.memory = None
        super().__init__(*args, **kwargs)
        self.temporal_padding = self.padding[0]
        self.memory_device = memory_device
        self.padding = (0, *self.padding[1:])  # Remove temporal pad to keep causal.
        self.memory_limit = float("inf")

    def set_memory_limit(self, value: float):
        self.memory_limit = value

    def set_memory_device(self, memory_device: _memory_device_t):
        self.memory_device = memory_device

    def memory_limit_conv(
        self,
        x,
        *,
        split_dim=3,
        padding=(0, 0, 0, 0, 0, 0),
        prev_cache=None,
    ):
        # Compatible with no limit.
        if math.isinf(self.memory_limit):
            if prev_cache is not None:
                x = torch.cat([prev_cache, x], dim=split_dim - 1)
            return super().forward(x)

        # Compute tensor shape after concat & padding.
        shape = torch.tensor(x.size())
        if prev_cache is not None:
            shape[split_dim - 1] += prev_cache.size(split_dim - 1)
        shape[-3:] += torch.tensor(padding).view(3, 2).sum(-1).flip(0)
        memory_occupy = shape.prod() * x.element_size() / 1024**3  # GiB
        logger.debug(f"x:{(shape, x.dtype)} {memory_occupy:.3f}GiB prev_cache:{prev_cache.shape if prev_cache is not None else None}")
        if memory_occupy < self.memory_limit or split_dim == x.ndim:
            if prev_cache is not None:
                x = torch.cat([prev_cache, x], dim=split_dim - 1)
            x = safe_pad_operation(x, padding, value=0.0)
            with ignore_padding(self):
                return super().forward(x)

        logger.debug(f"Exceed memory limit {memory_occupy} > {self.memory_limit}, split dim {split_dim}")

        # Split input (& prev_cache).
        num_splits = math.ceil(memory_occupy / self.memory_limit)
        size_per_split = x.size(split_dim) // num_splits
        split_sizes = [size_per_split] * (num_splits - 1)
        split_sizes += [x.size(split_dim) - sum(split_sizes)]

        x = list(x.split(split_sizes, dim=split_dim))
        logger.debug(f"Conv inputs: {[inp.size() for inp in x]} {x[0].dtype}")
        if prev_cache is not None:
            prev_cache = list(prev_cache.split(split_sizes, dim=split_dim))

        # Loop Fwd.
        cache = None
        for idx in range(len(x)):
            # Concat prev cache from last dim
            if prev_cache is not None:
                x[idx] = torch.cat([prev_cache[idx], x[idx]], dim=split_dim - 1)

            # Get padding pattern.
            lpad_dim = (x[idx].ndim - split_dim - 1) * 2
            rpad_dim = lpad_dim + 1
            padding = list(padding)
            padding[lpad_dim] = self.padding[split_dim - 2] if idx == 0 else 0
            padding[rpad_dim] = self.padding[split_dim - 2] if idx == len(x) - 1 else 0
            pad_len = padding[lpad_dim] + padding[rpad_dim]
            padding = tuple(padding)

            # Prepare cache for next slice (this dim).
            next_cache = None
            cache_len = cache.size(split_dim) if cache is not None else 0
            next_catch_size = get_cache_size(
                conv_module=self,
                input_len=x[idx].size(split_dim) + cache_len,
                pad_len=pad_len,
                dim=split_dim - 2,
            )
            if next_catch_size != 0:
                assert next_catch_size <= x[idx].size(split_dim)
                next_cache = x[idx].transpose(0, split_dim)[-next_catch_size:].transpose(0, split_dim)

            # Recursive.
            x[idx] = self.memory_limit_conv(
                x[idx],
                split_dim=split_dim + 1,
                padding=padding,
                prev_cache=cache,
            )

            # Update cache.
            cache = next_cache

        logger.debug(f"Conv outputs, concat(dim={split_dim}): {[d.size() for d in x]}")
        return torch.cat(x, split_dim)

    def forward(
        self,
        input: Union[Tensor, List[Tensor]],
        memory_state: MemoryState = MemoryState.UNSET,
    ) -> Tensor:
        assert memory_state != MemoryState.UNSET
        if memory_state != MemoryState.ACTIVE:
            self.memory = None
        if math.isinf(self.memory_limit) and torch.is_tensor(input) and get_sequence_parallel_group() is None:
            return self.basic_forward(input, memory_state)
        return self.slicing_forward(input, memory_state)

    def basic_forward(self, input: Tensor, memory_state: MemoryState = MemoryState.UNSET):
        mem_size = self.stride[0] - self.kernel_size[0]
        if (self.memory is not None) and (memory_state == MemoryState.ACTIVE):
            input = extend_head(input, memory=self.memory, times=-1)
        else:
            input = extend_head(input, times=self.temporal_padding * 2)
        memory = input[:, :, mem_size:].detach() if (mem_size != 0 and memory_state != MemoryState.DISABLED) else None
        if memory_state != MemoryState.DISABLED and not self.training and (self.memory_device is not None):
            self.memory = memory
            if self.memory_device == "cpu" and self.memory is not None:
                self.memory = self.memory.to("cpu")
        return super().forward(input)

    def slicing_forward(
        self,
        input: Union[Tensor, List[Tensor]],
        memory_state: MemoryState = MemoryState.UNSET,
    ) -> Tensor:
        squeeze_out = False
        if torch.is_tensor(input):
            input = [input]
            squeeze_out = True

        cache_size = self.kernel_size[0] - self.stride[0]
        cache = cache_send_recv(input, cache_size=cache_size, memory=self.memory, times=self.temporal_padding * 2)

        # For slice=4 and sp=2, and 17 frames in total
        #                  sp0                  sp1
        # slice 0: [`0 0` 0 1 2 {3 4}]   [{3 4} 5 6 (7 8)]    extend=`0 0` cache={3 4} memory=(7 8)
        # slice 1: [(7 8) 9 10 {11 12}]  [{11 12} 13 14 15 16]
        sp_rank = get_sequence_parallel_rank()
        sp_size = get_sequence_parallel_world_size()
        sp_group = get_sequence_parallel_group()
        send_dst = 0
        recv_src = 0
        if (
            memory_state in [MemoryState.INITIALIZING, MemoryState.ACTIVE]  # use_slicing
            and not self.training
            and (self.memory_device is not None)
            and sp_rank in [0, sp_size - 1]
            and cache_size != 0
        ):
            if cache_size > input[-1].size(2) and cache is not None and len(input) == 1:
                input[0] = torch.cat([cache, input[0]], dim=2)
                cache = None
            assert cache_size <= input[-1].size(2)
            if sp_size == 1:
                self.memory = input[-1][:, :, -cache_size:].detach().contiguous()
            else:
                if sp_rank == sp_size - 1:
                    dist.send(
                        input[-1][:, :, -cache_size:].detach().contiguous(),
                        send_dst,
                        group=sp_group,
                    )
                if sp_rank == 0:
                    shape = list(input[0].size())
                    shape[2] = cache_size
                    self.memory = torch.empty(*shape, device=input[0].device, dtype=input[0].dtype).contiguous()
                    dist.recv(self.memory, recv_src, group=sp_group)
            if self.memory_device == "cpu" and self.memory is not None:
                self.memory = self.memory.to("cpu")

        padding = tuple(x for x in reversed(self.padding) for _ in range(2))
        for i in range(len(input)):
            # Prepare cache for next input slice.
            next_cache = None
            cache_size = 0
            if i < len(input) - 1:
                cache_len = cache.size(2) if cache is not None else 0
                cache_size = get_cache_size(self, input[i].size(2) + cache_len, pad_len=0)
            if cache_size != 0:
                if cache_size > input[i].size(2) and cache is not None:
                    input[i] = torch.cat([cache, input[i]], dim=2)
                    cache = None
                assert cache_size <= input[i].size(2), f"{cache_size} > {input[i].size(2)}"
                next_cache = input[i][:, :, -cache_size:]

            # Conv forward for this input slice.
            input[i] = self.memory_limit_conv(
                input[i],
                padding=padding,
                prev_cache=cache,
            )

            # Update cache.
            cache = next_cache

        return input[0] if squeeze_out else input

    def tflops(self, args, kwargs, output) -> float:
        if torch.is_tensor(output):
            output_numel = output.numel()
        elif isinstance(output, list):
            output_numel = sum(o.numel() for o in output)
        else:
            raise NotImplementedError
        return (2 * math.prod(self.kernel_size) * self.in_channels * (output_numel / 1e6)) / 1e6

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if self.inflation_mode != "none":
            state_dict = modify_state_dict(
                self,
                state_dict,
                prefix,
                inflate_weight_fn=inflate_weight,
                inflate_bias_fn=inflate_bias,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            (strict and self.inflation_mode == "none"),
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def init_causal_conv3d(
    *args,
    inflation_mode: _inflation_mode_t,
    **kwargs,
):
    """
    Initialize a Causal-3D convolution layer.
    Parameters:
        inflation_mode: Listed as below. It's compatible with all the 3D-VAE checkpoints we have.
            - none: No inflation will be conducted.
                    The loading logic of state dict will fall back to default.
            - tail / replicate: Refer to the definition of `InflatedCausalConv3d`.
    """
    return InflatedCausalConv3d(*args, inflation_mode=inflation_mode, **kwargs)


def causal_norm_wrapper(norm_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    input_dtype = x.dtype
    if isinstance(norm_layer, (nn.LayerNorm, RMSNorm)):
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b h w c")
            x = norm_layer(x)
            x = rearrange(x, "b h w c -> b c h w")
            return x.to(input_dtype)
        if x.ndim == 5:
            x = rearrange(x, "b c t h w -> b t h w c")
            x = norm_layer(x)
            x = rearrange(x, "b t h w c -> b c t h w")
            return x.to(input_dtype)
    if isinstance(norm_layer, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
        if x.ndim <= 4:
            return norm_layer(x).to(input_dtype)
        if x.ndim == 5:
            t = x.size(2)
            x = rearrange(x, "b c t h w -> (b t) c h w")
            memory_occupy = x.numel() * x.element_size() / 1024**3
            if isinstance(norm_layer, nn.GroupNorm) and memory_occupy > get_norm_limit():
                num_chunks = min(4 if x.element_size() == 2 else 2, norm_layer.num_groups)
                logger.debug(f"large tensor {x.shape}, norm in {num_chunks} chunks")
                assert norm_layer.num_groups % num_chunks == 0
                num_groups_per_chunk = norm_layer.num_groups // num_chunks

                x = list(x.chunk(num_chunks, dim=1))
                weights = norm_layer.weight.chunk(num_chunks, dim=0)
                biases = norm_layer.bias.chunk(num_chunks, dim=0)
                for i, (w, b) in enumerate(zip(weights, biases)):
                    x[i] = F.group_norm(x[i], num_groups_per_chunk, w, b, norm_layer.eps)
                    x[i] = x[i].to(input_dtype)
                x = torch.cat(x, dim=1)
            else:
                x = norm_layer(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            return x.to(input_dtype)
    raise NotImplementedError


def remove_head(tensor: Tensor, times: int = 1) -> Tensor:
    """
    Remove duplicated first frame features in the up-sampling process.
    """
    sp_rank = get_sequence_parallel_rank()
    if times == 0 or sp_rank > 0:
        return tensor
    return torch.cat(tensors=(tensor[:, :, :1], tensor[:, :, times + 1 :]), dim=2)


def extend_head(tensor: Tensor, times: int = 2, memory: Optional[Tensor] = None) -> Tensor:
    """
    When memory is None:
        - Duplicate first frame features in the down-sampling process.
    When memory is not None:
        - Concatenate memory features with the input features to keep temporal consistency.
    """
    if memory is not None:
        return torch.cat((memory.to(tensor), tensor), dim=2)
    assert times >= 0, "Invalid input for function 'extend_head'!"
    if times == 0:
        return tensor
    else:
        tile_repeat = [1] * tensor.ndim
        tile_repeat[2] = times
        return torch.cat(tensors=(torch.tile(tensor[:, :, :1], tile_repeat), tensor), dim=2)


def inflate_weight(weight_2d: torch.Tensor, weight_3d: torch.Tensor, inflation_mode: str):
    """
    Inflate a 2D convolution weight matrix to a 3D one.
    Parameters:
        weight_2d:      The weight matrix of 2D conv to be inflated.
        weight_3d:      The weight matrix of 3D conv to be initialized.
        inflation_mode: the mode of inflation
    """
    assert inflation_mode in ["tail", "replicate"]
    assert weight_3d.shape[:2] == weight_2d.shape[:2]
    with torch.no_grad():
        if inflation_mode == "replicate":
            depth = weight_3d.size(2)
            weight_3d.copy_(weight_2d.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth)
        else:
            weight_3d.fill_(0.0)
            weight_3d[:, :, -1].copy_(weight_2d)
    return weight_3d


def inflate_bias(bias_2d: torch.Tensor, bias_3d: torch.Tensor, inflation_mode: str):
    """
    Inflate a 2D convolution bias tensor to a 3D one
    Parameters:
        bias_2d:        The bias tensor of 2D conv to be inflated.
        bias_3d:        The bias tensor of 3D conv to be initialized.
        inflation_mode: Placeholder to align `inflate_weight`.
    """
    assert bias_3d.shape == bias_2d.shape
    with torch.no_grad():
        bias_3d.copy_(bias_2d)
    return bias_3d


def modify_state_dict(layer, state_dict, prefix, inflate_weight_fn, inflate_bias_fn):
    """
    the main function to inflated 2D parameters to 3D.
    """
    weight_name = prefix + "weight"
    bias_name = prefix + "bias"
    if weight_name in state_dict:
        weight_2d = state_dict[weight_name]
        if weight_2d.dim() == 4:
            # Assuming the 2D weights are 4D tensors (out_channels, in_channels, h, w)
            weight_3d = inflate_weight_fn(
                weight_2d=weight_2d,
                weight_3d=layer.weight,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[weight_name] = weight_3d
        else:
            return state_dict
            # It's a 3d state dict, should not do inflation on both bias and weight.
    if bias_name in state_dict:
        bias_2d = state_dict[bias_name]
        if bias_2d.dim() == 1:
            # Assuming the 2D biases are 1D tensors (out_channels,)
            bias_3d = inflate_bias_fn(
                bias_2d=bias_2d,
                bias_3d=layer.bias,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[bias_name] = bias_3d
    return state_dict
