from abc import ABCMeta, abstractmethod
from typing import Literal

import torch

from lightx2v_platform.base.global_var import AI_DEVICE

FusedMoEActivation = Literal["gelu", "swiglu"]


class FusedMoETemplate(metaclass=ABCMeta):
    def __init__(self):
        self._parameters = {}

    def register_parameter(self, name, parameter):
        self._parameters[name] = parameter
        setattr(self, name, parameter)

    def named_parameters(self, prefix=""):
        for name, parameter in self._parameters.items():
            if parameter is not None:
                yield prefix + name, parameter

    def to_cpu(self, non_blocking=False):
        for name, parameter in self._parameters.items():
            if parameter is not None:
                self._parameters[name] = parameter.to("cpu", non_blocking=non_blocking)
                setattr(self, name, self._parameters[name])

    def to_cuda(self, non_blocking=False):
        for name, parameter in self._parameters.items():
            if parameter is not None:
                self._parameters[name] = parameter.to(AI_DEVICE, non_blocking=non_blocking)
                setattr(self, name, self._parameters[name])

    def to_cpu_async(self, non_blocking=True):
        self.to_cpu(non_blocking=non_blocking)

    def to_cuda_async(self, non_blocking=True):
        self.to_cuda(non_blocking=non_blocking)

    def state_dict(self, destination=None):
        return {} if destination is None else destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        return {} if destination is None else destination

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        pass

    @abstractmethod
    def apply(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def normalize_fused_moe_inputs(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input.ndim != 2:
        raise ValueError(f"fused MoE input must have shape [num_tokens, hidden_size], got {tuple(input.shape)}")
    if token_selected_experts.ndim != 2:
        raise ValueError(f"token_selected_experts must have shape [num_tokens, top_k], got {tuple(token_selected_experts.shape)}")
    if token_final_scales.shape != token_selected_experts.shape:
        raise ValueError(f"token_final_scales and token_selected_experts must have the same shape, got {tuple(token_final_scales.shape)} and {tuple(token_selected_experts.shape)}")
    if token_selected_experts.shape[0] != input.shape[0]:
        raise ValueError(f"routing tensors must have the same number of tokens as input, got input={input.shape[0]} and routing={token_selected_experts.shape[0]}")
    if token_selected_experts.device != input.device or token_final_scales.device != input.device:
        raise ValueError(f"input and routing tensors must be on the same device, got input={input.device}, experts={token_selected_experts.device}, scales={token_final_scales.device}")

    normalized_input = input if input.is_contiguous() else input.contiguous()
    normalized_experts = token_selected_experts.to(dtype=torch.int32).contiguous()
    normalized_scales = token_final_scales.to(dtype=torch.float32).contiguous()
    return normalized_input, normalized_experts, normalized_scales


def validate_fused_moe_output(
    output: torch.Tensor | None,
    shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if output is None:
        return
    if tuple(output.shape) != shape:
        raise ValueError(f"output must have shape {shape}, got {tuple(output.shape)}")
    if output.dtype != dtype:
        raise TypeError(f"output must have dtype {dtype}, got {output.dtype}")
    if output.device != device:
        raise ValueError(f"output must be on {device}, got {output.device}")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")


def validate_packed_local_experts(
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    activation: FusedMoEActivation,
    fc1_bias: torch.Tensor | None,
    fc2_bias: torch.Tensor | None,
) -> None:
    if activation not in {"gelu", "swiglu"}:
        raise ValueError(f"unsupported local fused MoE activation {activation!r}")
    if fc1_weight.ndim != 3:
        raise ValueError(f"fc1_weight must have shape [num_experts, out_features, in_features], got {tuple(fc1_weight.shape)}")
    if fc2_weight.ndim != 3:
        raise ValueError(f"fc2_weight must have shape [num_experts, out_features, in_features], got {tuple(fc2_weight.shape)}")
    if fc1_weight.shape[0] == 0:
        raise ValueError("local fused MoE requires at least one expert")
    if fc1_weight.shape[0] != fc2_weight.shape[0]:
        raise ValueError("fc1_weight and fc2_weight must have the same number of experts")
    intermediate_size = fc2_weight.shape[2]
    expected_fc1_size = intermediate_size if activation == "gelu" else 2 * intermediate_size
    if fc1_weight.shape[1] != expected_fc1_size:
        raise ValueError(f"{activation} fc1 output size must be {expected_fc1_size}, got {fc1_weight.shape[1]}")
    if fc1_weight.device != fc2_weight.device:
        raise ValueError("fc1_weight and fc2_weight must be on the same device")
    if fc1_weight.dtype != fc2_weight.dtype:
        raise TypeError("fc1_weight and fc2_weight must have the same dtype")
    if not fc1_weight.is_floating_point():
        raise TypeError(f"local fused MoE weights must be floating point, got {fc1_weight.dtype}")
    for name, bias, shape in (
        ("fc1_bias", fc1_bias, (fc1_weight.shape[0], fc1_weight.shape[1])),
        ("fc2_bias", fc2_bias, (fc2_weight.shape[0], fc2_weight.shape[1])),
    ):
        if bias is None:
            continue
        if tuple(bias.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(bias.shape)}")
        if bias.device != fc1_weight.device:
            raise ValueError(f"{name} and expert weights must be on the same device")
        if bias.dtype != fc1_weight.dtype:
            raise TypeError(f"{name} and expert weights must have the same dtype")


__all__ = ["FusedMoEActivation", "FusedMoETemplate"]
