"""Registered fused MoE backends."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import FUSED_MOE_REGISTER

try:
    from flashinfer.fused_moe import cutlass_fused_moe as _flashinfer_cutlass_fused_moe
except Exception:  # Some FlashInfer builds fail during import.
    try:
        import flashinfer

        _flashinfer_cutlass_fused_moe = flashinfer.fused_moe.cutlass_fused_moe
    except Exception:
        _flashinfer_cutlass_fused_moe = None

try:
    from flashinfer.tllm_enums import ActivationType as _FlashInferActivationType
except Exception:
    _FlashInferActivationType = None

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


FusedMoEActivation = Literal["gelu", "swiglu"]

_MULTI_MICRO_SHARDS = 2
_MULTI_MICRO_NUM_EXPERTS = 64
_MULTI_MICRO_TOP_K = 8
_MULTI_MICRO_HIDDEN_SIZE = 4096
_MULTI_MICRO_INTERMEDIATE_SIZE = 768
_MULTI_MICRO_FC1_SIZE = 2 * _MULTI_MICRO_INTERMEDIATE_SIZE


if _TRITON_AVAILABLE:

    @triton.jit
    def _invert_permutation_kernel(
        permuted_to_expanded,
        expanded_to_permuted,
        num_routes,
        BLOCK_SIZE: tl.constexpr,
    ):
        permuted_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        active = permuted_idx < num_routes
        expanded_idx = tl.load(permuted_to_expanded + permuted_idx, mask=active)
        tl.store(expanded_to_permuted + expanded_idx, permuted_idx, mask=active)

    @triton.jit
    def _multi_micro_finalize_kernel(
        permuted_expert_output_0,
        permuted_expert_output_1,
        expanded_to_permuted,
        routing_scales,
        output,
        hidden_size: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        hidden_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        hidden_mask = hidden_idx < hidden_size
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        # Sum both TP partials in FP32 before applying routing weights.
        for route_idx in tl.static_range(0, TOP_K):
            expanded_idx = token_idx * TOP_K + route_idx
            permuted_idx = tl.load(expanded_to_permuted + expanded_idx)
            value_0 = tl.load(
                permuted_expert_output_0 + permuted_idx * hidden_size + hidden_idx,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            value_1 = tl.load(
                permuted_expert_output_1 + permuted_idx * hidden_size + hidden_idx,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            scale = tl.load(routing_scales + expanded_idx).to(tl.float32)
            accumulator += (value_0 + value_1) * scale

        tl.store(output + token_idx * hidden_size + hidden_idx, accumulator, mask=hidden_mask)


class FusedMoETemplate(WeightModule, metaclass=ABCMeta):
    @abstractmethod
    def apply(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def _normalize_inputs(
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


def _validate_output(
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


@dataclass(frozen=True)
class FlashInferMoEWeightShard:
    fc1_weight: torch.Tensor
    fc2_weight: torch.Tensor
    tp_rank: int = 0
    fc1_bias: torch.Tensor | None = None
    fc2_bias: torch.Tensor | None = None
    quant_scales: Sequence[torch.Tensor] | None = None


def _flashinfer_result_tensor(result, output: torch.Tensor) -> torch.Tensor:
    if result is None:
        return output
    if torch.is_tensor(result):
        result_tensor = result
    elif isinstance(result, (tuple, list)) and result and torch.is_tensor(result[0]):
        result_tensor = result[0]
    else:
        raise RuntimeError(f"FlashInfer fused MoE returned an unsupported result of type {type(result)!r}")

    if result_tensor is not output:
        output.copy_(result_tensor)
    return output


@FUSED_MOE_REGISTER("flashinfer")
class FlashInferFusedMoE(FusedMoETemplate):
    def __init__(
        self,
        shards: FlashInferMoEWeightShard | Sequence[FlashInferMoEWeightShard],
        activation: FusedMoEActivation,
        tp_size: int = 1,
        output_dtype: torch.dtype | None = None,
        tune_max_num_tokens: int = 8192,
    ):
        super().__init__()
        if isinstance(shards, FlashInferMoEWeightShard):
            shards = (shards,)
        else:
            shards = tuple(shards)
        if not shards:
            raise ValueError("FlashInfer fused MoE requires at least one weight shard")
        if any(not isinstance(shard, FlashInferMoEWeightShard) for shard in shards):
            raise TypeError("shards must contain only FlashInferMoEWeightShard instances")
        if activation not in {"gelu", "swiglu"}:
            raise ValueError(f"unsupported FlashInfer fused MoE activation {activation!r}; expected 'gelu' or 'swiglu'")
        if tp_size < 1:
            raise ValueError(f"tp_size must be positive, got {tp_size}")
        for shard in shards:
            if shard.tp_rank < 0 or shard.tp_rank >= tp_size:
                raise ValueError(f"FlashInfer TP rank {shard.tp_rank} is outside TP size {tp_size}")
            if shard.fc2_weight.ndim < 2:
                raise ValueError(f"FlashInfer fc2_weight must have at least two dimensions, got {tuple(shard.fc2_weight.shape)}")
        output_size = shards[0].fc2_weight.shape[1]
        if any(shard.fc2_weight.shape[1] != output_size for shard in shards[1:]):
            raise ValueError("all FlashInfer weight shards must have the same FC2 output size")
        if tune_max_num_tokens < 1:
            raise ValueError(f"tune_max_num_tokens must be positive, got {tune_max_num_tokens}")

        shard_specs = []
        for shard_index, shard in enumerate(shards):
            prefix = f"shard_{shard_index}"
            self.register_parameter(f"{prefix}_fc1_weight", shard.fc1_weight)
            self.register_parameter(f"{prefix}_fc2_weight", shard.fc2_weight)
            self.register_parameter(f"{prefix}_fc1_bias", shard.fc1_bias)
            self.register_parameter(f"{prefix}_fc2_bias", shard.fc2_bias)
            if shard.quant_scales is None:
                quant_scale_names = None
            else:
                quant_scale_names = tuple(f"{prefix}_quant_scale_{index}" for index in range(len(shard.quant_scales)))
                for name, scale in zip(quant_scale_names, shard.quant_scales, strict=True):
                    self.register_parameter(name, scale)
            shard_specs.append((prefix, shard.tp_rank, quant_scale_names))
        self._shard_specs = tuple(shard_specs)
        self.activation = activation
        self.tp_size = int(tp_size)
        self.output_dtype = output_dtype
        self.tune_max_num_tokens = int(tune_max_num_tokens)
        self.output_size = int(output_size)

    @property
    def shards(self) -> tuple[FlashInferMoEWeightShard, ...]:
        shards = []
        for prefix, tp_rank, quant_scale_names in self._shard_specs:
            quant_scales = None if quant_scale_names is None else tuple(getattr(self, name) for name in quant_scale_names)
            shards.append(
                FlashInferMoEWeightShard(
                    getattr(self, f"{prefix}_fc1_weight"),
                    getattr(self, f"{prefix}_fc2_weight"),
                    tp_rank=tp_rank,
                    fc1_bias=getattr(self, f"{prefix}_fc1_bias"),
                    fc2_bias=getattr(self, f"{prefix}_fc2_bias"),
                    quant_scales=quant_scales,
                )
            )
        return tuple(shards)

    @staticmethod
    def is_available() -> bool:
        return _flashinfer_cutlass_fused_moe is not None and _FlashInferActivationType is not None

    def _activation_type(self):
        if _FlashInferActivationType is None:
            raise ImportError("FlashInfer activation type is unavailable")
        if self.activation == "gelu":
            return _FlashInferActivationType.Gelu
        return _FlashInferActivationType.Swiglu

    def _apply_shard(
        self,
        shard: FlashInferMoEWeightShard,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        output: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        quant_scales = None if shard.quant_scales is None else list(shard.quant_scales)
        result = _flashinfer_cutlass_fused_moe(
            input,
            token_selected_experts,
            token_final_scales,
            shard.fc1_weight,
            shard.fc2_weight,
            output_dtype,
            quant_scales=quant_scales,
            fc1_expert_biases=shard.fc1_bias,
            fc2_expert_biases=shard.fc2_bias,
            tp_size=self.tp_size,
            tp_rank=shard.tp_rank,
            output=output,
            tune_max_num_tokens=self.tune_max_num_tokens,
            activation_type=self._activation_type(),
        )
        return _flashinfer_result_tensor(result, output)

    def apply(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _flashinfer_cutlass_fused_moe is None:
            raise ImportError("FlashInfer fused MoE is unavailable")

        input, token_selected_experts, token_final_scales = _normalize_inputs(input, token_selected_experts, token_final_scales)
        output_dtype = input.dtype if self.output_dtype is None else self.output_dtype
        output_shape = (input.shape[0], self.output_size)
        _validate_output(output, shape=output_shape, dtype=output_dtype, device=input.device)
        combined_output = output if output is not None else torch.empty(output_shape, dtype=output_dtype, device=input.device)

        shards = self.shards
        self._apply_shard(
            shards[0],
            input,
            token_selected_experts,
            token_final_scales,
            combined_output,
            output_dtype,
        )
        for shard in shards[1:]:
            shard_output = torch.empty_like(combined_output)
            self._apply_shard(
                shard,
                input,
                token_selected_experts,
                token_final_scales,
                shard_output,
                output_dtype,
            )
            combined_output.add_(shard_output)
        return combined_output


@lru_cache(maxsize=None)
def _validate_multi_micro_device(device_index: int) -> None:
    capability = torch.cuda.get_device_capability(device_index)
    if capability != (9, 0):
        name = torch.cuda.get_device_name(device_index)
        raise RuntimeError(f"lightx2v_multi_micro_fused_moe only supports SM90 Hopper GPUs; device cuda:{device_index} is {name!r} with capability {capability}")


def _require_multi_micro_tensor(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_multi_micro_inputs(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_micro_weights: torch.Tensor,
    fc2_micro_weights: torch.Tensor,
    output: torch.Tensor | None,
    backend: str,
) -> torch.Tensor:
    if backend != "grouped_mm":
        raise ValueError(f"unsupported multi-micro MoE backend {backend!r}; expected 'grouped_mm'")
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("this PyTorch build does not provide torch._grouped_mm; the Hopper grouped_mm backend is unavailable")
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must have dtype torch.bfloat16, got {input.dtype}")
    if input.ndim != 2 or input.shape[1] != _MULTI_MICRO_HIDDEN_SIZE:
        raise ValueError(f"input must have shape [num_tokens, {_MULTI_MICRO_HIDDEN_SIZE}], got {tuple(input.shape)}")
    if input.shape[0] == 0:
        raise ValueError("input must contain at least one token")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")

    device = input.device
    _validate_multi_micro_device(device.index if device.index is not None else torch.cuda.current_device())
    num_tokens = input.shape[0]
    _require_multi_micro_tensor(
        token_selected_experts,
        "token_selected_experts",
        shape=(num_tokens, _MULTI_MICRO_TOP_K),
        dtype=torch.int32,
        device=device,
    )
    _require_multi_micro_tensor(
        token_final_scales,
        "token_final_scales",
        shape=(num_tokens, _MULTI_MICRO_TOP_K),
        dtype=torch.float32,
        device=device,
    )
    _require_multi_micro_tensor(
        fc1_micro_weights,
        "fc1_micro_weights",
        shape=(
            _MULTI_MICRO_SHARDS,
            _MULTI_MICRO_NUM_EXPERTS,
            _MULTI_MICRO_FC1_SIZE,
            _MULTI_MICRO_HIDDEN_SIZE,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    _require_multi_micro_tensor(
        fc2_micro_weights,
        "fc2_micro_weights",
        shape=(
            _MULTI_MICRO_SHARDS,
            _MULTI_MICRO_NUM_EXPERTS,
            _MULTI_MICRO_HIDDEN_SIZE,
            _MULTI_MICRO_INTERMEDIATE_SIZE,
        ),
        dtype=torch.bfloat16,
        device=device,
    )

    if output is None:
        return torch.empty_like(input)
    _require_multi_micro_tensor(
        output,
        "output",
        shape=(num_tokens, _MULTI_MICRO_HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    return output


def _finalize_multi_micro_with_triton(
    permuted_expert_output_0: torch.Tensor,
    permuted_expert_output_1: torch.Tensor,
    permuted_to_expanded: torch.Tensor,
    token_final_scales: torch.Tensor,
    output: torch.Tensor,
) -> None:
    num_routes = permuted_to_expanded.numel()
    expanded_to_permuted = torch.empty_like(permuted_to_expanded)
    inverse_block = 256
    _invert_permutation_kernel[(triton.cdiv(num_routes, inverse_block),)](
        permuted_to_expanded,
        expanded_to_permuted,
        num_routes,
        BLOCK_SIZE=inverse_block,
        num_warps=4,
    )

    finalize_block = 256
    grid = (output.shape[0], triton.cdiv(_MULTI_MICRO_HIDDEN_SIZE, finalize_block))
    _multi_micro_finalize_kernel[grid](
        permuted_expert_output_0,
        permuted_expert_output_1,
        expanded_to_permuted,
        token_final_scales,
        output,
        hidden_size=_MULTI_MICRO_HIDDEN_SIZE,
        TOP_K=_MULTI_MICRO_TOP_K,
        BLOCK_SIZE=finalize_block,
        num_warps=4,
    )


def _finalize_multi_micro_with_torch(
    permuted_expert_output_0: torch.Tensor,
    permuted_expert_output_1: torch.Tensor,
    permuted_to_expanded: torch.Tensor,
    token_final_scales: torch.Tensor,
    output: torch.Tensor,
) -> None:
    permuted_expert_output = permuted_expert_output_0.float()
    permuted_expert_output.add_(permuted_expert_output_1.float())
    expanded_output = torch.empty_like(permuted_expert_output)
    expanded_output.index_copy_(0, permuted_to_expanded, permuted_expert_output)
    weighted = expanded_output.view(-1, _MULTI_MICRO_TOP_K, _MULTI_MICRO_HIDDEN_SIZE)
    weighted.mul_(token_final_scales.unsqueeze(-1))
    output.copy_(weighted.sum(dim=1))


def lightx2v_multi_micro_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_micro_weights: torch.Tensor,
    fc2_micro_weights: torch.Tensor,
    output: torch.Tensor | None = None,
    backend: str = "grouped_mm",
) -> torch.Tensor:
    """Two-shard, 64-expert, top-8 SM90 BF16 SwiGLU MoE for HunyuanImage-3."""

    output = _validate_multi_micro_inputs(
        input,
        token_selected_experts,
        token_final_scales,
        fc1_micro_weights,
        fc2_micro_weights,
        output,
        backend,
    )

    # Reuse the routing permutation for both micro shards.
    flat_experts = token_selected_experts.reshape(-1)
    permuted_to_expanded = torch.argsort(flat_experts)
    permuted_token_indices = torch.div(permuted_to_expanded, _MULTI_MICRO_TOP_K, rounding_mode="floor")
    permuted_input = input.index_select(0, permuted_token_indices)
    expert_offsets = torch.bincount(flat_experts, minlength=_MULTI_MICRO_NUM_EXPERTS).cumsum(0).to(torch.int32)

    micro_activations: list[torch.Tensor] = []
    for micro_idx in range(_MULTI_MICRO_SHARDS):
        projected = torch._grouped_mm(
            permuted_input,
            fc1_micro_weights[micro_idx].transpose(1, 2),
            offs=expert_offsets,
        )
        gate, up = projected.chunk(2, dim=-1)
        # Hunyuan packs gate before up: gate * silu(up).
        activated = F.silu(up)
        activated.mul_(gate)
        micro_activations.append(activated)

    permuted_expert_outputs = (
        torch._grouped_mm(
            micro_activations[0],
            fc2_micro_weights[0].transpose(1, 2),
            offs=expert_offsets,
        ),
        torch._grouped_mm(
            micro_activations[1],
            fc2_micro_weights[1].transpose(1, 2),
            offs=expert_offsets,
        ),
    )

    if _TRITON_AVAILABLE:
        _finalize_multi_micro_with_triton(
            permuted_expert_outputs[0],
            permuted_expert_outputs[1],
            permuted_to_expanded,
            token_final_scales,
            output,
        )
    else:  # pragma: no cover
        _finalize_multi_micro_with_torch(
            permuted_expert_outputs[0],
            permuted_expert_outputs[1],
            permuted_to_expanded,
            token_final_scales,
            output,
        )
    return output


@FUSED_MOE_REGISTER("multi_micro")
class MultiMicroFusedMoE(FusedMoETemplate):
    def __init__(
        self,
        fc1_micro_weights: torch.Tensor,
        fc2_micro_weights: torch.Tensor,
        activation: FusedMoEActivation = "swiglu",
        grouped_backend: str = "grouped_mm",
    ):
        super().__init__()
        if activation != "swiglu":
            raise ValueError(f"multi_micro fused MoE supports only 'swiglu', got {activation!r}")
        self.register_parameter("fc1_micro_weights", fc1_micro_weights)
        self.register_parameter("fc2_micro_weights", fc2_micro_weights)
        self.activation = activation
        self.grouped_backend = grouped_backend

    def apply(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input, token_selected_experts, token_final_scales = _normalize_inputs(input, token_selected_experts, token_final_scales)
        return lightx2v_multi_micro_fused_moe(
            input,
            token_selected_experts,
            token_final_scales,
            self.fc1_micro_weights,
            self.fc2_micro_weights,
            output=output,
            backend=self.grouped_backend,
        )


def _torch_expert_weights(
    weights: torch.Tensor | Sequence[torch.Tensor],
    name: str,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
    if torch.is_tensor(weights):
        if weights.ndim != 3:
            raise ValueError(f"{name} must have shape [num_experts, out_features, in_features], got {tuple(weights.shape)}")
        if weights.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one expert")
        return tuple(weights.unbind(0)), weights

    expert_weights = tuple(weights)
    if not expert_weights:
        raise ValueError(f"{name} must contain at least one expert")
    if any(not torch.is_tensor(weight) for weight in expert_weights):
        raise TypeError(f"{name} must contain only tensors")
    if any(weight.ndim != 2 for weight in expert_weights):
        shapes = [tuple(weight.shape) for weight in expert_weights]
        raise ValueError(f"{name} expert weights must be two-dimensional, got {shapes}")
    return expert_weights, None


def _torch_expert_biases(
    biases: torch.Tensor | Sequence[torch.Tensor] | None,
    name: str,
    num_experts: int,
) -> tuple[tuple[torch.Tensor | None, ...], torch.Tensor | None]:
    if biases is None:
        return (None,) * num_experts, None
    if torch.is_tensor(biases):
        if biases.ndim != 2 or biases.shape[0] != num_experts:
            raise ValueError(f"{name} must have shape [num_experts, out_features], got {tuple(biases.shape)}")
        return tuple(biases.unbind(0)), biases

    expert_biases = tuple(biases)
    if len(expert_biases) != num_experts:
        raise ValueError(f"{name} must contain {num_experts} expert biases, got {len(expert_biases)}")
    if any(not torch.is_tensor(bias) for bias in expert_biases):
        raise TypeError(f"{name} must contain only tensors")
    if any(bias.ndim != 1 for bias in expert_biases):
        shapes = [tuple(bias.shape) for bias in expert_biases]
        raise ValueError(f"{name} expert biases must be one-dimensional, got {shapes}")
    return expert_biases, None


class _TorchFusedMoE(FusedMoETemplate):
    def __init__(
        self,
        fc1_weight: torch.Tensor | Sequence[torch.Tensor],
        fc2_weight: torch.Tensor | Sequence[torch.Tensor],
        activation: FusedMoEActivation,
        grouped: bool,
        fc1_bias: torch.Tensor | Sequence[torch.Tensor] | None = None,
        fc2_bias: torch.Tensor | Sequence[torch.Tensor] | None = None,
        fc1_gate_weight: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ):
        super().__init__()
        if activation not in {"gelu", "swiglu"}:
            raise ValueError(f"unsupported Torch fused MoE activation {activation!r}; expected 'gelu' or 'swiglu'")
        if fc1_gate_weight is not None and fc1_bias is not None:
            raise ValueError("split SwiGLU does not support fc1_bias")

        fc1_weights, packed_fc1_weight = _torch_expert_weights(fc1_weight, "fc1_weight")
        fc2_weights, packed_fc2_weight = _torch_expert_weights(fc2_weight, "fc2_weight")
        if fc1_gate_weight is None:
            fc1_gate_weights = ()
            packed_fc1_gate_weight = None
        else:
            if activation != "swiglu":
                raise ValueError("fc1_gate_weight is valid only with swiglu activation")
            fc1_gate_weights, packed_fc1_gate_weight = _torch_expert_weights(fc1_gate_weight, "fc1_gate_weight")
        num_experts = len(fc1_weights)
        if len(fc2_weights) != num_experts:
            raise ValueError(f"fc1_weight and fc2_weight must have the same number of experts, got {num_experts} and {len(fc2_weights)}")
        if fc1_gate_weights and len(fc1_gate_weights) != num_experts:
            raise ValueError(f"fc1_weight and fc1_gate_weight must have the same number of experts, got {num_experts} and {len(fc1_gate_weights)}")

        fc1_shape = tuple(fc1_weights[0].shape)
        fc2_shape = tuple(fc2_weights[0].shape)
        if any(tuple(weight.shape) != fc1_shape for weight in fc1_weights):
            raise ValueError("all fc1 expert weights must have the same shape")
        if any(tuple(weight.shape) != fc2_shape for weight in fc2_weights):
            raise ValueError("all fc2 expert weights must have the same shape")
        if fc1_gate_weights and any(tuple(weight.shape) != fc1_shape for weight in fc1_gate_weights):
            raise ValueError("fc1_gate_weight must have the same expert shape as fc1_weight")
        fc1_size, input_size = fc1_shape
        output_size, intermediate_size = fc2_shape
        split_swiglu = bool(fc1_gate_weights)
        expected_fc1_size = intermediate_size if activation == "gelu" or split_swiglu else 2 * intermediate_size
        if fc1_size != expected_fc1_size:
            raise ValueError(f"{activation} fc1 output size must be {expected_fc1_size}, got {fc1_size}")

        weight_device = fc1_weights[0].device
        weight_dtype = fc1_weights[0].dtype
        if not fc1_weights[0].is_floating_point():
            raise TypeError(f"Torch fused MoE weights must be floating point, got {weight_dtype}")
        weight_layers = [("fc1_weight", fc1_weights), ("fc2_weight", fc2_weights)]
        if split_swiglu:
            weight_layers.append(("fc1_gate_weight", fc1_gate_weights))
        for name, weights_for_layer in weight_layers:
            for weight in weights_for_layer:
                if weight.device != weight_device:
                    raise ValueError(f"all Torch fused MoE weights must be on {weight_device}, got {name} on {weight.device}")
                if weight.dtype != weight_dtype:
                    raise TypeError(f"all Torch fused MoE weights must have dtype {weight_dtype}, got {name} with {weight.dtype}")

        fc1_biases, packed_fc1_bias = _torch_expert_biases(fc1_bias, "fc1_bias", num_experts)
        fc2_biases, packed_fc2_bias = _torch_expert_biases(fc2_bias, "fc2_bias", num_experts)
        for name, biases_for_layer, size in (
            ("fc1_bias", fc1_biases, fc1_size),
            ("fc2_bias", fc2_biases, output_size),
        ):
            for bias in biases_for_layer:
                if bias is None:
                    continue
                if tuple(bias.shape) != (size,):
                    raise ValueError(f"{name} expert biases must have shape ({size},), got {tuple(bias.shape)}")
                if bias.device != weight_device:
                    raise ValueError(f"{name} must be on {weight_device}, got {bias.device}")
                if bias.dtype != weight_dtype:
                    raise TypeError(f"{name} must have dtype {weight_dtype}, got {bias.dtype}")

        self.activation = activation
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self._grouped = grouped
        self.split_swiglu = split_swiglu
        self._needs_permuted_experts = grouped and (fc1_biases[0] is not None or fc2_biases[0] is not None)

        if grouped:
            grouped_fc1_weight = (packed_fc1_weight if packed_fc1_weight is not None else torch.stack(fc1_weights, dim=0)).contiguous()
            grouped_fc2_weight = (packed_fc2_weight if packed_fc2_weight is not None else torch.stack(fc2_weights, dim=0)).contiguous()
            grouped_fc1_gate_weight = None if not split_swiglu else (packed_fc1_gate_weight if packed_fc1_gate_weight is not None else torch.stack(fc1_gate_weights, dim=0)).contiguous()
            grouped_fc1_bias = None if fc1_biases[0] is None else (packed_fc1_bias if packed_fc1_bias is not None else torch.stack(fc1_biases, dim=0)).contiguous()
            grouped_fc2_bias = None if fc2_biases[0] is None else (packed_fc2_bias if packed_fc2_bias is not None else torch.stack(fc2_biases, dim=0)).contiguous()
            self.register_parameter("grouped_fc1_weight", grouped_fc1_weight)
            self.register_parameter("grouped_fc2_weight", grouped_fc2_weight)
            self.register_parameter("grouped_fc1_gate_weight", grouped_fc1_gate_weight)
            self.register_parameter("grouped_fc1_bias", grouped_fc1_bias)
            self.register_parameter("grouped_fc2_bias", grouped_fc2_bias)
            self._fc1_weight_names = ()
            self._fc2_weight_names = ()
            self._fc1_gate_weight_names = ()
            self._fc1_bias_names = ()
            self._fc2_bias_names = ()
        else:
            self._fc1_weight_names = self._register_expert_tensors("fc1_weight", fc1_weights)
            self._fc2_weight_names = self._register_expert_tensors("fc2_weight", fc2_weights)
            self._fc1_gate_weight_names = self._register_expert_tensors("fc1_gate_weight", fc1_gate_weights)
            self._fc1_bias_names = self._register_expert_tensors("fc1_bias", fc1_biases)
            self._fc2_bias_names = self._register_expert_tensors("fc2_bias", fc2_biases)

    def _register_expert_tensors(
        self,
        prefix: str,
        tensors: Sequence[torch.Tensor | None],
    ) -> tuple[str | None, ...]:
        names = []
        for expert_index, tensor in enumerate(tensors):
            if tensor is None:
                names.append(None)
                continue
            name = f"{prefix}_{expert_index}"
            self.register_parameter(name, tensor)
            names.append(name)
        return tuple(names)

    def _registered_expert_tensors(self, names: Sequence[str | None]) -> tuple[torch.Tensor | None, ...]:
        return tuple(None if name is None else getattr(self, name) for name in names)

    @property
    def fc1_weights(self):
        return self._registered_expert_tensors(self._fc1_weight_names)

    @property
    def fc2_weights(self):
        return self._registered_expert_tensors(self._fc2_weight_names)

    @property
    def fc1_gate_weights(self):
        return self._registered_expert_tensors(self._fc1_gate_weight_names)

    @property
    def fc1_biases(self):
        return self._registered_expert_tensors(self._fc1_bias_names)

    @property
    def fc2_biases(self):
        return self._registered_expert_tensors(self._fc2_bias_names)

    @property
    def weight_device(self):
        if self._grouped:
            return self.grouped_fc1_weight.device
        return self.fc1_weights[0].device

    @property
    def weight_dtype(self):
        if self._grouped:
            return self.grouped_fc1_weight.dtype
        return self.fc1_weights[0].dtype

    def _activate(self, projected: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        if self.activation == "gelu":
            return F.gelu(projected, approximate="none")
        if self.split_swiglu:
            assert gate is not None
            return projected * F.silu(gate)
        value, gate = projected.chunk(2, dim=-1)
        # Packed SwiGLU stores value first and the SiLU gate second.
        return value * F.silu(gate)

    def _permute_routes(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        top_k = token_selected_experts.shape[1]
        flat_experts = token_selected_experts.reshape(-1).to(torch.int64)
        counts = torch.zeros(self.num_experts, dtype=torch.int64, device=input.device)
        counts.scatter_add_(0, flat_experts, torch.ones_like(flat_experts))
        permuted_to_expanded = torch.argsort(flat_experts)
        permuted_experts = flat_experts.index_select(0, permuted_to_expanded) if self._needs_permuted_experts else None
        token_indices = torch.div(permuted_to_expanded, top_k, rounding_mode="floor")
        if input.device.type == "mlu":
            permuted_to_expanded = permuted_to_expanded.to(torch.int32)
            token_indices = token_indices.to(torch.int32)
        permuted_input = input.index_select(0, token_indices)
        return permuted_input, permuted_to_expanded, permuted_experts, counts

    def _run_experts(
        self,
        permuted_input: torch.Tensor,
        permuted_experts: torch.Tensor | None,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def apply(
        self,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_selected_experts.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
            raise TypeError(f"token_selected_experts must have an integer dtype, got {token_selected_experts.dtype}")
        if not token_final_scales.is_floating_point():
            raise TypeError(f"token_final_scales must have a floating-point dtype, got {token_final_scales.dtype}")
        input, token_selected_experts, token_final_scales = _normalize_inputs(input, token_selected_experts, token_final_scales)
        if input.shape[1] != self.input_size:
            raise ValueError(f"input hidden size must be {self.input_size}, got {input.shape[1]}")
        if input.device != self.weight_device:
            raise ValueError(f"input and Torch fused MoE weights must be on the same device, got input={input.device} and weights={self.weight_device}")
        if input.dtype != self.weight_dtype:
            raise TypeError(f"input and Torch fused MoE weights must have the same dtype, got input={input.dtype} and weights={self.weight_dtype}")

        top_k = token_selected_experts.shape[1]
        if top_k < 1 or top_k > self.num_experts:
            raise ValueError(f"top_k must be between 1 and {self.num_experts}, got {top_k}")
        output_shape = (input.shape[0], self.output_size)
        _validate_output(output, shape=output_shape, dtype=input.dtype, device=input.device)
        result = output if output is not None else torch.empty(output_shape, dtype=input.dtype, device=input.device)
        if input.shape[0] == 0:
            return result

        permuted_input, permuted_to_expanded, permuted_experts, counts = self._permute_routes(input, token_selected_experts)
        permuted_output = self._run_experts(permuted_input, permuted_experts, counts)
        expanded_output = torch.empty_like(permuted_output)
        expanded_output.index_copy_(0, permuted_to_expanded, permuted_output)
        route_output = expanded_output.view(input.shape[0], top_k, self.output_size)
        accumulate_dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
        routed = torch.bmm(
            token_final_scales.to(accumulate_dtype).unsqueeze(1),
            route_output.to(accumulate_dtype),
        ).squeeze(1)
        result.copy_(routed)
        return result


@FUSED_MOE_REGISTER("torch_grouped_mm")
class TorchGroupedMMFusedMoE(_TorchFusedMoE):
    def __init__(
        self,
        fc1_weight: torch.Tensor | Sequence[torch.Tensor],
        fc2_weight: torch.Tensor | Sequence[torch.Tensor],
        activation: FusedMoEActivation,
        fc1_bias: torch.Tensor | Sequence[torch.Tensor] | None = None,
        fc2_bias: torch.Tensor | Sequence[torch.Tensor] | None = None,
        fc1_gate_weight: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ):
        super().__init__(
            fc1_weight,
            fc2_weight,
            activation=activation,
            fc1_bias=fc1_bias,
            fc2_bias=fc2_bias,
            grouped=True,
            fc1_gate_weight=fc1_gate_weight,
        )

    def _run_experts(
        self,
        permuted_input: torch.Tensor,
        permuted_experts: torch.Tensor | None,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(torch, "_grouped_mm"):
            raise RuntimeError("this PyTorch build does not provide torch._grouped_mm")
        if not permuted_input.is_cuda:
            raise ValueError("torch_grouped_mm requires CUDA tensors")

        offsets = counts.cumsum(0, dtype=torch.int32)
        projected = torch._grouped_mm(
            permuted_input,
            self.grouped_fc1_weight.transpose(1, 2),
            offs=offsets,
        )
        # The CUDA grouped_mm kernel currently rejects its public bias argument.
        if self.grouped_fc1_bias is not None:
            assert permuted_experts is not None
            projected.add_(self.grouped_fc1_bias.index_select(0, permuted_experts))
        gate = None
        if self.split_swiglu:
            gate = torch._grouped_mm(
                permuted_input,
                self.grouped_fc1_gate_weight.transpose(1, 2),
                offs=offsets,
            )
        projected = self._activate(projected, gate)
        expert_output = torch._grouped_mm(
            projected,
            self.grouped_fc2_weight.transpose(1, 2),
            offs=offsets,
        )
        if self.grouped_fc2_bias is not None:
            assert permuted_experts is not None
            expert_output.add_(self.grouped_fc2_bias.index_select(0, permuted_experts))
        return expert_output


@FUSED_MOE_REGISTER("torch_expert_loop")
class TorchExpertLoopFusedMoE(_TorchFusedMoE):
    def __init__(
        self,
        fc1_weight: torch.Tensor | Sequence[torch.Tensor],
        fc2_weight: torch.Tensor | Sequence[torch.Tensor],
        activation: FusedMoEActivation,
        fc1_bias: torch.Tensor | Sequence[torch.Tensor] | None = None,
        fc2_bias: torch.Tensor | Sequence[torch.Tensor] | None = None,
        fc1_gate_weight: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ):
        super().__init__(
            fc1_weight,
            fc2_weight,
            activation=activation,
            fc1_bias=fc1_bias,
            fc2_bias=fc2_bias,
            grouped=False,
            fc1_gate_weight=fc1_gate_weight,
        )

    def _run_experts(
        self,
        permuted_input: torch.Tensor,
        permuted_experts: torch.Tensor | None,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        permuted_output = torch.empty(
            (permuted_input.shape[0], self.output_size),
            dtype=permuted_input.dtype,
            device=permuted_input.device,
        )
        start = 0
        for expert_idx, count in enumerate(counts.cpu().tolist()):
            end = start + count
            if count:
                expert_input = permuted_input[start:end]
                fc1_bias = self.fc1_biases[expert_idx]
                if fc1_bias is None:
                    projected = torch.mm(expert_input, self.fc1_weights[expert_idx].t())
                else:
                    projected = torch.addmm(fc1_bias, expert_input, self.fc1_weights[expert_idx].t())
                gate = None
                if self.split_swiglu:
                    gate = torch.mm(expert_input, self.fc1_gate_weights[expert_idx].t())
                projected = self._activate(projected, gate)
                destination = permuted_output[start:end]
                fc2_bias = self.fc2_biases[expert_idx]
                if fc2_bias is None:
                    torch.mm(projected, self.fc2_weights[expert_idx].t(), out=destination)
                else:
                    torch.addmm(fc2_bias, projected, self.fc2_weights[expert_idx].t(), out=destination)
            start = end
        return permuted_output


def _stack_local_experts(tensors):
    if torch.is_tensor(tensors):
        return tensors.contiguous()
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError("local fused MoE tensor sequence must not be empty")
    return torch.stack(tensors, dim=0)


def _validate_packed_local_experts(
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    activation: FusedMoEActivation,
    fc1_bias: torch.Tensor | None,
    fc2_bias: torch.Tensor | None,
):
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


def create_local_fused_moe(
    backend: Literal["flashinfer", "npu_grouped_mm", "metax_mctlass_moe", "torch_grouped_mm", "torch_expert_loop"],
    fc1_weight: torch.Tensor | Sequence[torch.Tensor],
    fc2_weight: torch.Tensor | Sequence[torch.Tensor],
    activation: FusedMoEActivation,
    fc1_bias: torch.Tensor | Sequence[torch.Tensor] | None = None,
    fc2_bias: torch.Tensor | Sequence[torch.Tensor] | None = None,
    tune_max_num_tokens: int = 8192,
    fc1_gate_weight: torch.Tensor | Sequence[torch.Tensor] | None = None,
) -> FusedMoETemplate:
    if backend in {"torch_expert_loop", "torch_grouped_mm"}:
        return FUSED_MOE_REGISTER[backend](
            fc1_weight,
            fc2_weight,
            activation,
            fc1_bias,
            fc2_bias,
            fc1_gate_weight,
        )
    if backend not in {"flashinfer", "npu_grouped_mm", "metax_mctlass_moe"}:
        raise ValueError(f"unsupported local fused MoE backend {backend!r}")
    if fc1_gate_weight is not None:
        raise ValueError("split SwiGLU weights are supported only by Torch fused MoE backends")

    packed_fc1_weight = _stack_local_experts(fc1_weight)
    packed_fc2_weight = _stack_local_experts(fc2_weight)
    packed_fc1_bias = None if fc1_bias is None else _stack_local_experts(fc1_bias)
    packed_fc2_bias = None if fc2_bias is None else _stack_local_experts(fc2_bias)
    _validate_packed_local_experts(
        packed_fc1_weight,
        packed_fc2_weight,
        activation,
        packed_fc1_bias,
        packed_fc2_bias,
    )
    if backend in {"npu_grouped_mm", "metax_mctlass_moe"}:
        return FUSED_MOE_REGISTER[backend](
            packed_fc1_weight,
            packed_fc2_weight,
            activation,
            packed_fc1_bias,
            packed_fc2_bias,
        )

    shard = FlashInferMoEWeightShard(
        packed_fc1_weight,
        packed_fc2_weight,
        fc1_bias=packed_fc1_bias,
        fc2_bias=packed_fc2_bias,
    )
    return FUSED_MOE_REGISTER[backend](
        shard,
        activation,
        tp_size=1,
        tune_max_num_tokens=tune_max_num_tokens,
    )


__all__ = [
    "FlashInferFusedMoE",
    "FlashInferMoEWeightShard",
    "FusedMoEActivation",
    "FusedMoETemplate",
    "MultiMicroFusedMoE",
    "TorchExpertLoopFusedMoE",
    "TorchGroupedMMFusedMoE",
    "create_local_fused_moe",
    "lightx2v_multi_micro_fused_moe",
]
