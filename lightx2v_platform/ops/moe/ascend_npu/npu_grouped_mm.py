from importlib import import_module

import torch
import torch.nn.functional as F

from lightx2v_platform.ops.moe.template import (
    FusedMoEActivation,
    FusedMoETemplate,
    normalize_fused_moe_inputs,
    validate_fused_moe_output,
    validate_packed_local_experts,
)
from lightx2v_platform.registry_factory import PLATFORM_FUSED_MOE_REGISTER

_TORCH_NPU = None
_REQUIRED_NPU_MOE_OPS = (
    "npu_moe_init_routing_v2",
    "npu_grouped_matmul",
    "npu_moe_token_unpermute",
)


def _load_torch_npu():
    global _TORCH_NPU
    if _TORCH_NPU is None:
        try:
            _TORCH_NPU = import_module("torch_npu")
        except (ImportError, RuntimeError) as error:
            raise RuntimeError("npu_grouped_mm fused MoE requires torch_npu") from error

    missing = [name for name in _REQUIRED_NPU_MOE_OPS if not hasattr(_TORCH_NPU, name)]
    if missing:
        raise RuntimeError(f"npu_grouped_mm fused MoE requires missing torch_npu ops: {', '.join(missing)}")
    return _TORCH_NPU


@PLATFORM_FUSED_MOE_REGISTER("npu_grouped_mm")
class NpuGroupedMMFusedMoE(FusedMoETemplate):
    """Native Ascend fused MoE using routing and grouped-matmul operators."""

    def __init__(
        self,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        activation: FusedMoEActivation,
        fc1_bias: torch.Tensor | None = None,
        fc2_bias: torch.Tensor | None = None,
    ):
        super().__init__()
        if activation != "gelu":
            raise ValueError(f"unsupported NPU grouped-MM fused MoE activation {activation!r}; expected 'gelu'")

        validate_packed_local_experts(
            fc1_weight,
            fc2_weight,
            activation,
            fc1_bias,
            fc2_bias,
        )
        if fc1_weight.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise TypeError(f"NPU grouped-MM fused MoE supports float16, bfloat16, or float32 weights, got {fc1_weight.dtype}")

        # The common layout is [experts, out_features, in_features], while
        # npu_grouped_matmul consumes [experts, in_features, out_features].
        self.register_parameter("grouped_fc1_weight", fc1_weight.transpose(1, 2).contiguous())
        self.register_parameter("grouped_fc2_weight", fc2_weight.transpose(1, 2).contiguous())

        # aclnnGroupedMatmul requires FP32 bias for BF16 inputs and weights.
        bias_dtype = torch.float32 if fc1_weight.dtype == torch.bfloat16 else fc1_weight.dtype
        grouped_fc1_bias = None if fc1_bias is None else fc1_bias.to(dtype=bias_dtype).contiguous()
        grouped_fc2_bias = None if fc2_bias is None else fc2_bias.to(dtype=bias_dtype).contiguous()
        self.register_parameter("grouped_fc1_bias", grouped_fc1_bias)
        self.register_parameter("grouped_fc2_bias", grouped_fc2_bias)

        self.activation = activation
        self.num_experts = int(fc1_weight.shape[0])
        self.input_size = int(fc1_weight.shape[2])
        self.output_size = int(fc2_weight.shape[1])

    @property
    def weight_device(self):
        return self.grouped_fc1_weight.device

    @property
    def weight_dtype(self):
        return self.grouped_fc1_weight.dtype

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

        input, token_selected_experts, token_final_scales = normalize_fused_moe_inputs(input, token_selected_experts, token_final_scales)
        if input.device.type != "npu":
            raise ValueError(f"npu_grouped_mm fused MoE requires NPU input, got {input.device}")
        if input.shape[1] != self.input_size:
            raise ValueError(f"input hidden size must be {self.input_size}, got {input.shape[1]}")
        if input.device != self.weight_device:
            raise ValueError(f"input and NPU grouped-MM fused MoE weights must be on the same device, got input={input.device} and weights={self.weight_device}")
        if input.dtype != self.weight_dtype:
            raise TypeError(f"input and NPU grouped-MM fused MoE weights must have the same dtype, got input={input.dtype} and weights={self.weight_dtype}")

        top_k = token_selected_experts.shape[1]
        if top_k < 1 or top_k > self.num_experts:
            raise ValueError(f"top_k must be between 1 and {self.num_experts}, got {top_k}")
        output_shape = (input.shape[0], self.output_size)
        validate_fused_moe_output(output, shape=output_shape, dtype=input.dtype, device=input.device)
        if input.shape[0] == 0:
            return output if output is not None else torch.empty(output_shape, dtype=input.dtype, device=input.device)

        torch_npu = _load_torch_npu()
        # npu_moe_token_unpermute expects routing probabilities in the same
        # floating-point dtype as the expert output.
        token_final_scales = token_final_scales.to(dtype=input.dtype).contiguous()
        expanded_input, expanded_row_idx, expert_counts, _ = torch_npu.npu_moe_init_routing_v2(
            input,
            token_selected_experts,
            active_num=token_selected_experts.numel(),
            expert_num=self.num_experts,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            quant_mode=-1,
            active_expert_range=[0, self.num_experts],
            row_idx_type=0,
        )

        fc1_bias = None if self.grouped_fc1_bias is None else [self.grouped_fc1_bias]
        hidden = torch_npu.npu_grouped_matmul(
            x=[expanded_input],
            weight=[self.grouped_fc1_weight],
            bias=fc1_bias,
            group_list=expert_counts,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]
        hidden = F.gelu(hidden, approximate="none")

        fc2_bias = None if self.grouped_fc2_bias is None else [self.grouped_fc2_bias]
        expert_output = torch_npu.npu_grouped_matmul(
            x=[hidden],
            weight=[self.grouped_fc2_weight],
            bias=fc2_bias,
            group_list=expert_counts,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]
        routed = torch_npu.npu_moe_token_unpermute(
            expert_output,
            torch.abs(expanded_row_idx),
            probs=token_final_scales,
        )
        if output is not None:
            output.copy_(routed)
            return output
        return routed


__all__ = ["NpuGroupedMMFusedMoE"]
