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


def _load_metax_moe_ops():
    if not torch.cuda.is_available():
        raise RuntimeError("metax_mctlass_moe requires an available MetaX CUDA device; check the MACA runtime or use moe_backend='torch_expert_loop'")
    try:
        import_module("mcoplib._moe_C")
        mctlass_ex = import_module("mctlassEx")
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError("metax_mctlass_moe requires the MetaX mcoplib and mctlassEx extensions; install matching MACA/PyTorch packages or use moe_backend='torch_expert_loop'") from error

    fused_moe_gemm = getattr(mctlass_ex, "FusedMoeGEMM", None)
    if fused_moe_gemm is None:
        raise RuntimeError("metax_mctlass_moe requires mctlassEx.FusedMoeGEMM; install a compatible mctlassEx package or use moe_backend='torch_expert_loop'")
    try:
        align_op = torch.ops._moe_C.moe_align_block_size
        sum_op = torch.ops._moe_C.moe_sum
    except AttributeError as error:
        raise RuntimeError("metax_mctlass_moe requires mcoplib _moe_C routing operators; install a compatible mcoplib package or use moe_backend='torch_expert_loop'") from error
    return fused_moe_gemm(), align_op, sum_op


@PLATFORM_FUSED_MOE_REGISTER("metax_mctlass_moe")
class MetaxMctlassFusedMoE(FusedMoETemplate):
    """MetaX BF16 local MoE built from MCTlass grouped GEMM kernels."""

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
            raise ValueError(f"unsupported MetaX MCTlass fused MoE activation {activation!r}; expected 'gelu'")

        validate_packed_local_experts(
            fc1_weight,
            fc2_weight,
            activation,
            fc1_bias,
            fc2_bias,
        )
        if fc1_weight.dtype != torch.bfloat16:
            raise TypeError(f"MetaX MCTlass fused MoE supports only torch.bfloat16 weights; got {fc1_weight.dtype}. Use a BF16 model or moe_backend='torch_expert_loop'")

        self.register_parameter("grouped_fc1_weight", fc1_weight.contiguous())
        self.register_parameter("grouped_fc2_weight", fc2_weight.contiguous())
        self.register_parameter("grouped_fc1_bias", None if fc1_bias is None else fc1_bias.contiguous())
        self.register_parameter("grouped_fc2_bias", None if fc2_bias is None else fc2_bias.contiguous())

        self.activation = activation
        self.num_experts = int(fc1_weight.shape[0])
        self.input_size = int(fc1_weight.shape[2])
        self.intermediate_size = int(fc2_weight.shape[2])
        self.output_size = int(fc2_weight.shape[1])
        self._mctlass_moe_gemm, self._align_op, self._sum_op = _load_metax_moe_ops()
        self._kernel_block_sizes = {}

    @property
    def weight_device(self):
        return self.grouped_fc1_weight.device

    @property
    def weight_dtype(self):
        return self.grouped_fc1_weight.dtype

    def _get_kernel_block_size(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        top_k: int,
        mul_routed_weight: bool,
        stage: str,
    ) -> int:
        cache_key = (
            stage,
            input.shape[0],
            input.shape[1],
            weight.shape[1],
            top_k,
            input.dtype,
            input.device,
        )
        if cache_key in self._kernel_block_sizes:
            return self._kernel_block_sizes[cache_key]

        block_size = int(
            self._mctlass_moe_gemm.get_kernel_m(
                input,
                weight,
                output,
                self.num_experts,
                input.shape[0],
                weight.shape[1],
                input.shape[1],
                top_k,
                mul_weight=mul_routed_weight,
            )
        )
        if block_size <= 0:
            raise RuntimeError(f"MetaX MCTlass fused MoE found no {stage} kernel for the requested shape")
        self._kernel_block_sizes[cache_key] = block_size
        return block_size

    def _align_routes(
        self,
        token_selected_experts: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        route_count = token_selected_experts.numel()
        max_padded_routes = route_count + self.num_experts * (block_size - 1)
        if route_count < self.num_experts:
            max_padded_routes = min(route_count * block_size, max_padded_routes)

        sorted_token_ids = torch.empty(max_padded_routes, dtype=torch.int32, device=token_selected_experts.device)
        expert_ids = torch.empty(
            (max_padded_routes + block_size - 1) // block_size,
            dtype=torch.int32,
            device=token_selected_experts.device,
        )
        num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device=token_selected_experts.device)
        self._align_op(
            token_selected_experts,
            self.num_experts,
            block_size,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            None,
        )
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    def _run_grouped_gemm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        bias: torch.Tensor | None,
        routing_weights: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        top_k: int,
        mul_routed_weight: bool,
    ) -> None:
        self._mctlass_moe_gemm(
            input.shape[0],
            weight.shape[1],
            input.shape[1],
            self.num_experts,
            sorted_token_ids.numel(),
            top_k,
            input,
            weight,
            output,
            None,
            None,
            bias,
            routing_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
        )

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
        if not input.is_cuda:
            raise ValueError(f"MetaX MCTlass fused MoE requires CUDA input, got {input.device}")
        if input.shape[1] != self.input_size:
            raise ValueError(f"input hidden size must be {self.input_size}, got {input.shape[1]}")
        if input.device != self.weight_device:
            raise ValueError(f"input and MetaX MCTlass fused MoE weights must be on the same device, got input={input.device} and weights={self.weight_device}")
        if input.dtype != torch.bfloat16:
            raise TypeError(f"MetaX MCTlass fused MoE supports only torch.bfloat16 input; got {input.dtype}. Use a BF16 model or moe_backend='torch_expert_loop'")

        top_k = token_selected_experts.shape[1]
        if top_k < 1 or top_k > self.num_experts:
            raise ValueError(f"top_k must be between 1 and {self.num_experts}, got {top_k}")
        output_shape = (input.shape[0], self.output_size)
        validate_fused_moe_output(output, shape=output_shape, dtype=input.dtype, device=input.device)
        result = output if output is not None else torch.empty(output_shape, dtype=input.dtype, device=input.device)
        if input.shape[0] == 0:
            return result

        stage1_output = torch.empty(
            (input.shape[0], top_k, self.intermediate_size),
            dtype=input.dtype,
            device=input.device,
        )
        stage1_block_size = self._get_kernel_block_size(
            input,
            self.grouped_fc1_weight,
            stage1_output,
            top_k,
            False,
            "stage-1",
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = self._align_routes(token_selected_experts, stage1_block_size)
        self._run_grouped_gemm(
            input,
            self.grouped_fc1_weight,
            stage1_output,
            self.grouped_fc1_bias,
            token_final_scales,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            False,
        )

        hidden = F.gelu(stage1_output, approximate="none").reshape(-1, self.intermediate_size).contiguous()
        del stage1_output
        stage2_output = torch.empty(
            (input.shape[0], top_k, self.output_size),
            dtype=input.dtype,
            device=input.device,
        )
        stage2_block_size = self._get_kernel_block_size(
            hidden,
            self.grouped_fc2_weight,
            stage2_output,
            1,
            True,
            "stage-2",
        )
        if stage2_block_size != stage1_block_size:
            sorted_token_ids, expert_ids, num_tokens_post_padded = self._align_routes(token_selected_experts, stage2_block_size)
        self._run_grouped_gemm(
            hidden,
            self.grouped_fc2_weight,
            stage2_output,
            self.grouped_fc2_bias,
            token_final_scales.reshape(-1, 1),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            True,
        )
        self._sum_op(stage2_output, result)
        return result


__all__ = ["MetaxMctlassFusedMoE"]
