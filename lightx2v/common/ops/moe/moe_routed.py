"""Routed-expert MoE backends.

Routing policy and shared experts intentionally stay in each model.  These
operators consume an already selected route (or, for the Ascend fused path,
router logits) and return only the routed-expert contribution.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MOE_ROUTED_REGISTER


ExpertFn = Callable[[Any, int, torch.Tensor], torch.Tensor]
GroupedExpertFn = Callable[[Any, torch.Tensor, torch.Tensor], torch.Tensor]


def _validate_selected_routes(hidden_states, topk_idx, topk_weight):
    if hidden_states.ndim < 2:
        raise ValueError(f"MoE hidden_states must have at least 2 dimensions, got {tuple(hidden_states.shape)}")

    num_tokens = hidden_states.numel() // hidden_states.shape[-1]
    if topk_idx.ndim != 2 or topk_weight.ndim != 2:
        raise ValueError(f"MoE top-k tensors must be rank 2, got indices={tuple(topk_idx.shape)}, weights={tuple(topk_weight.shape)}")
    if topk_idx.shape != topk_weight.shape:
        raise ValueError(f"MoE top-k shape mismatch: indices={tuple(topk_idx.shape)}, weights={tuple(topk_weight.shape)}")
    if topk_idx.shape[0] != num_tokens:
        raise ValueError(f"MoE route token count mismatch: hidden_states has {num_tokens} tokens, routes have {topk_idx.shape[0]}")
    if topk_idx.shape[1] == 0:
        raise ValueError("MoE top-k must be positive")
    if topk_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"MoE top-k indices must be int32 or int64, got {topk_idx.dtype}")
    if topk_idx.device != hidden_states.device or topk_weight.device != hidden_states.device:
        raise ValueError(
            f"MoE routes must be on {hidden_states.device}, got indices={topk_idx.device}, weights={topk_weight.device}"
        )


@MOE_ROUTED_REGISTER("torch")
class TorchMoeRouted(WeightModule):
    """Dispatch, execute, and combine routed experts with PyTorch.

    The model supplies its expert math through ``expert_fn`` or
    ``grouped_expert_fn``.  This keeps GELU/SwiGLU layouts, grouped-matmul
    padding, tensor parallelism, and quantized weight wrappers out of the
    common routing operator.
    """

    @torch.no_grad()
    def apply(
        self,
        hidden_states,
        topk_idx,
        topk_weight,
        *,
        num_experts,
        expert_context=None,
        expert_fn: ExpertFn | None = None,
        grouped_expert_fn: GroupedExpertFn | None = None,
        accumulate_dtype: torch.dtype | None = None,
        drop_zero_routes=False,
        route_index_dtype: torch.dtype | None = None,
        combine_mode="index_add",
    ):
        _validate_selected_routes(hidden_states, topk_idx, topk_weight)
        if num_experts <= 0:
            raise ValueError(f"MoE num_experts must be positive, got {num_experts}")
        if expert_fn is None and grouped_expert_fn is None:
            raise ValueError("Torch routed MoE requires expert_fn or grouped_expert_fn")

        original_shape = hidden_states.shape
        hidden_dim = original_shape[-1]
        flat = hidden_states.reshape(-1, hidden_dim)
        top_k = topk_idx.shape[1]
        flat_indices = topk_idx.reshape(-1)
        flat_weights = topk_weight.reshape(-1)

        if drop_zero_routes:
            active_positions = torch.where(flat_weights != 0)[0]
        else:
            active_positions = torch.arange(flat_indices.numel(), device=flat_indices.device, dtype=torch.long)

        if active_positions.numel() == 0:
            return torch.zeros_like(hidden_states)

        active_experts = flat_indices[active_positions]
        counts = torch.zeros(num_experts, device=flat_indices.device, dtype=torch.int64)
        try:
            counts.scatter_add_(0, active_experts.to(torch.int64), torch.ones_like(active_experts, dtype=torch.int64))
        except RuntimeError as error:
            raise ValueError(f"MoE expert index is outside [0, {num_experts})") from error
        sort_order = torch.argsort(active_experts, stable=True)
        sorted_positions = active_positions[sort_order]
        if route_index_dtype is not None:
            sorted_positions = sorted_positions.to(route_index_dtype)
        token_indices = torch.div(sorted_positions, top_k, rounding_mode="floor")
        permuted_tokens = flat[token_indices]

        if grouped_expert_fn is not None:
            expert_output = grouped_expert_fn(expert_context, permuted_tokens, counts)
        else:
            outputs = []
            for expert_idx, expert_tokens in enumerate(torch.split(permuted_tokens, counts.tolist(), dim=0)):
                if expert_tokens.numel() != 0:
                    outputs.append(expert_fn(expert_context, expert_idx, expert_tokens))
            expert_output = torch.cat(outputs, dim=0) if outputs else flat.new_empty((0, hidden_dim))

        expected_shape = (sorted_positions.numel(), hidden_dim)
        if tuple(expert_output.shape) != expected_shape:
            raise ValueError(f"MoE expert executor returned {tuple(expert_output.shape)}, expected {expected_shape}")

        sorted_weights = flat_weights[sorted_positions]
        if combine_mode == "index_add":
            combine_dtype = expert_output.dtype if accumulate_dtype is None else accumulate_dtype
            weighted = expert_output.to(combine_dtype)
            weighted.mul_(sorted_weights.to(combine_dtype).unsqueeze(-1))
            combined = torch.zeros((flat.shape[0], hidden_dim), dtype=combine_dtype, device=expert_output.device)
            combined.index_add_(0, token_indices.to(torch.int64), weighted)
            combined = combined.to(expert_output.dtype)
        elif combine_mode == "slot_sum":
            expanded_output = expert_output.new_zeros((flat_indices.numel(), hidden_dim))
            expanded_output[sorted_positions] = expert_output
            expanded_output = expanded_output.view(flat.shape[0], top_k, hidden_dim)
            route_weight = topk_weight.unsqueeze(-1)
            if accumulate_dtype is None:
                combined = (expanded_output * route_weight.to(expanded_output.dtype)).sum(dim=1)
            else:
                combined = (expanded_output.to(accumulate_dtype) * route_weight.to(accumulate_dtype)).sum(dim=1).to(expert_output.dtype)
        else:
            raise ValueError(f"Unsupported MoE combine_mode={combine_mode!r}; expected 'index_add' or 'slot_sum'")
        return combined.reshape(original_shape)


_FLASHINFER_FUSED_MOE = None
_FLASHINFER_ACTIVATION_TYPE = None


def _load_flashinfer_fused_moe():
    global _FLASHINFER_ACTIVATION_TYPE, _FLASHINFER_FUSED_MOE
    if _FLASHINFER_FUSED_MOE is not None:
        return _FLASHINFER_FUSED_MOE, _FLASHINFER_ACTIVATION_TYPE

    try:
        from flashinfer.fused_moe import cutlass_fused_moe
    except Exception:
        try:
            flashinfer = importlib.import_module("flashinfer")
            cutlass_fused_moe = flashinfer.fused_moe.cutlass_fused_moe
        except Exception as second_error:
            raise RuntimeError("moe_backend='flashinfer' requires flashinfer.fused_moe.cutlass_fused_moe") from second_error

    try:
        from flashinfer.tllm_enums import ActivationType
    except Exception:
        ActivationType = None

    _FLASHINFER_FUSED_MOE = cutlass_fused_moe
    _FLASHINFER_ACTIVATION_TYPE = ActivationType
    return _FLASHINFER_FUSED_MOE, _FLASHINFER_ACTIVATION_TYPE


def _flashinfer_activation_type(activation_type_enum, activation):
    if activation is None:
        return None
    if activation_type_enum is None:
        raise RuntimeError(f"FlashInfer activation_type={activation!r} requires flashinfer.tllm_enums.ActivationType")

    aliases = {
        "gelu": ("Gelu", "GELU"),
        "swiglu": ("Swiglu", "SwiGlu", "SWIGLU"),
        "silu": ("Silu", "SiLU", "SILU"),
    }
    activation = str(activation).strip().lower()
    if activation not in aliases:
        raise ValueError(f"Unsupported FlashInfer MoE activation {activation!r}")
    for attr in aliases[activation]:
        if hasattr(activation_type_enum, attr):
            return getattr(activation_type_enum, attr)
    raise RuntimeError(f"Installed FlashInfer does not expose an ActivationType for {activation!r}")


@MOE_ROUTED_REGISTER("flashinfer")
class FlashInferMoeRouted(WeightModule):
    """CUTLASS fused routed experts provided by FlashInfer."""

    @torch.no_grad()
    def apply(
        self,
        hidden_states,
        topk_idx,
        topk_weight,
        *,
        fc1_weight,
        fc2_weight,
        output_dtype=None,
        quant_scales=None,
        fc1_bias=None,
        fc2_bias=None,
        tune_max_num_tokens=8192,
        activation=None,
        output=None,
        tp_size=None,
        tp_rank=None,
    ):
        _validate_selected_routes(hidden_states, topk_idx, topk_weight)
        if hidden_states.device.type != "cuda":
            raise RuntimeError(f"moe_backend='flashinfer' requires CUDA inputs, got {hidden_states.device}")

        fused_moe, activation_type_enum = _load_flashinfer_fused_moe()
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1]).contiguous()
        output_flat = None if output is None else output.reshape_as(flat)
        kwargs = {
            "quant_scales": quant_scales,
            "fc1_expert_biases": fc1_bias,
            "fc2_expert_biases": fc2_bias,
            "tune_max_num_tokens": tune_max_num_tokens,
        }
        if output_flat is not None:
            kwargs["output"] = output_flat
        activation_type = _flashinfer_activation_type(activation_type_enum, activation)
        if activation_type is not None:
            kwargs["activation_type"] = activation_type
        if tp_size is not None:
            kwargs["tp_size"] = tp_size
        if tp_rank is not None:
            kwargs["tp_rank"] = tp_rank

        result = fused_moe(
            flat,
            topk_idx.to(torch.int32).contiguous(),
            topk_weight.to(torch.float32).contiguous(),
            fc1_weight,
            fc2_weight,
            output_dtype or flat.dtype,
            **kwargs,
        )
        routed = output_flat if output_flat is not None else (result[0] if isinstance(result, (tuple, list)) else result)
        return routed.reshape(original_shape)


_TORCH_NPU = None
_REQUIRED_NPU_MOE_OPS = (
    "npu_moe_gating_top_k_softmax",
    "npu_moe_init_routing_v2",
    "npu_grouped_matmul",
    "npu_moe_token_unpermute",
)


def _load_torch_npu():
    global _TORCH_NPU
    if _TORCH_NPU is None:
        try:
            _TORCH_NPU = importlib.import_module("torch_npu")
        except (ImportError, RuntimeError) as error:
            raise RuntimeError("moe_backend='npu' requires torch_npu") from error
    missing = [name for name in _REQUIRED_NPU_MOE_OPS if not hasattr(_TORCH_NPU, name)]
    if missing:
        raise RuntimeError(f"moe_backend='npu' requires missing torch_npu ops: {', '.join(missing)}")
    return _TORCH_NPU


@MOE_ROUTED_REGISTER("npu")
class NpuMoeRouted(WeightModule):
    """Native Ascend routed experts using routing and grouped-matmul ops."""

    @torch.no_grad()
    def apply(
        self,
        hidden_states,
        topk_idx=None,
        topk_weight=None,
        *,
        router_logits=None,
        top_k=None,
        num_experts,
        fc1_weight,
        fc2_weight,
        fc1_bias=None,
        fc2_bias=None,
        activation="gelu",
    ):
        torch_npu = _load_torch_npu()
        if hidden_states.device.type != "npu":
            raise RuntimeError(f"moe_backend='npu' requires NPU inputs, got {hidden_states.device}")

        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1]).contiguous()
        if (topk_idx is None) != (topk_weight is None):
            raise ValueError("NPU routed MoE requires topk_idx and topk_weight together")
        if fc1_weight.ndim != 3 or fc2_weight.ndim != 3:
            raise ValueError(f"NPU MoE weights must be rank 3, got fc1={tuple(fc1_weight.shape)}, fc2={tuple(fc2_weight.shape)}")
        if fc1_weight.device != flat.device or fc2_weight.device != flat.device:
            raise ValueError(f"NPU MoE weights must be on {flat.device}, got fc1={fc1_weight.device}, fc2={fc2_weight.device}")
        if fc1_weight.shape[0] != num_experts or fc2_weight.shape[0] != num_experts:
            raise ValueError(f"NPU MoE expert count mismatch: expected {num_experts}, got fc1={fc1_weight.shape[0]}, fc2={fc2_weight.shape[0]}")
        if fc1_weight.shape[1] != flat.shape[1] or fc1_weight.shape[2] != fc2_weight.shape[1] or fc2_weight.shape[2] != flat.shape[1]:
            raise ValueError(
                f"NPU MoE shape mismatch: input={tuple(flat.shape)}, fc1={tuple(fc1_weight.shape)}, fc2={tuple(fc2_weight.shape)}"
            )
        if flat.dtype != fc1_weight.dtype or flat.dtype != fc2_weight.dtype:
            raise ValueError(f"NPU MoE input/weight dtype mismatch: input={flat.dtype}, fc1={fc1_weight.dtype}, fc2={fc2_weight.dtype}")
        for name, bias in (("fc1_bias", fc1_bias), ("fc2_bias", fc2_bias)):
            if bias is not None and bias.device != flat.device:
                raise ValueError(f"NPU MoE {name} must be on {flat.device}, got {bias.device}")

        if topk_idx is None or topk_weight is None:
            if router_logits is None or top_k is None:
                raise ValueError("NPU routed MoE requires topk_idx/topk_weight or router_logits/top_k")
            if router_logits.device != flat.device:
                raise ValueError(f"NPU MoE router logits must be on {flat.device}, got {router_logits.device}")
            if router_logits.numel() != flat.shape[0] * num_experts:
                raise ValueError(
                    f"NPU MoE router logits must have {flat.shape[0] * num_experts} elements for [{flat.shape[0]}, {num_experts}], got {router_logits.numel()}"
                )
            if not 0 < top_k <= num_experts:
                raise ValueError(f"NPU MoE top_k must be in [1, {num_experts}], got {top_k}")
            router_logits = router_logits.reshape(flat.shape[0], num_experts)
            topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k_softmax(router_logits, None, k=top_k)
        else:
            _validate_selected_routes(flat, topk_idx, topk_weight)
            topk_idx = topk_idx.to(torch.int32).contiguous()
            topk_weight = topk_weight.contiguous()

        expanded_x, expanded_row_idx, expert_counts, _ = torch_npu.npu_moe_init_routing_v2(
            flat,
            topk_idx,
            active_num=topk_idx.numel(),
            expert_num=num_experts,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            quant_mode=-1,
            active_expert_range=[0, num_experts],
            row_idx_type=0,
        )

        grouped_fc1_bias = None if fc1_bias is None else [fc1_bias]
        hidden = torch_npu.npu_grouped_matmul(
            x=[expanded_x],
            weight=[fc1_weight],
            bias=grouped_fc1_bias,
            group_list=expert_counts,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]
        if activation == "gelu":
            hidden = F.gelu(hidden, approximate="none")
        else:
            raise NotImplementedError(f"NPU routed MoE activation {activation!r} is not supported")

        grouped_fc2_bias = None if fc2_bias is None else [fc2_bias]
        expert_output = torch_npu.npu_grouped_matmul(
            x=[hidden],
            weight=[fc2_weight],
            bias=grouped_fc2_bias,
            group_list=expert_counts,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]
        routed = torch_npu.npu_moe_token_unpermute(
            expert_output,
            torch.abs(expanded_row_idx),
            probs=topk_weight,
        )
        return routed.reshape(original_shape)
