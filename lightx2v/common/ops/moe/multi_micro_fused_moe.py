"""Hopper BF16 multi-micro-shard MoE inference operator.

This module implements a targeted H200 backend for HunyuanImage-3's
phase-dependent TP layout.  During denoising a logical TP2 expert weight is
stored as two TP4 micro shards::

    fc1: [micro=2, expert=64, 2 * intermediate=1536, hidden=4096]
    fc2: [micro=2, expert=64, hidden=4096, intermediate=768]

The public function deliberately consumes those tensors without stacking,
transposing into a new allocation, or otherwise reorganising their storage.
It creates one expert permutation and one set of expert offsets, reuses them
for both micro shards, sums the two partial GEMM2 results, and applies routing
scores exactly once in the final reduction.

FlashInfer's CUTLASS fused-MoE runner cannot directly express this layout: one
runner invocation creates one GEMM problem per expert and owns its routing and
finalize workspaces.  Calling it once per micro shard repeats both operations.
This implementation instead reuses one routing permutation across four Hopper
grouped GEMMs, then performs one Triton FP32 micro-sum/top-k finalize.  It is
one logical LightX2V operator call, not one CUDA kernel launch.

The implementation is intentionally narrow and fails fast outside its target:
CUDA SM90, BF16, two micro shards, 64 experts, top-k 8, SwiGLU, no bias,
quantization, LoRA, or expert parallelism.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_MICRO_SHARDS: Final = 2
_NUM_EXPERTS: Final = 64
_TOP_K: Final = 8
_HIDDEN_SIZE: Final = 4096
_INTERMEDIATE_SIZE: Final = 768
_FC1_SIZE: Final = 2 * _INTERMEDIATE_SIZE


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

        # Sum the two BF16 GEMM2 partials in FP32 immediately before applying
        # the router scale.  The score is therefore multiplied once,
        # independently of the number of micro shards.
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


@lru_cache(maxsize=None)
def _validate_device(device_index: int) -> None:
    capability = torch.cuda.get_device_capability(device_index)
    if capability != (9, 0):
        name = torch.cuda.get_device_name(device_index)
        raise RuntimeError(f"lightx2v_multi_micro_fused_moe only supports SM90 Hopper GPUs; device cuda:{device_index} is {name!r} with capability {capability}")


def _require_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
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


def _validate_inputs(
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
        raise RuntimeError("this PyTorch build does not provide torch._grouped_mm; the H200 grouped_mm backend is unavailable")
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must have dtype torch.bfloat16, got {input.dtype}")
    if input.ndim != 2 or input.shape[1] != _HIDDEN_SIZE:
        raise ValueError(f"input must have shape [num_tokens, {_HIDDEN_SIZE}], got {tuple(input.shape)}")
    if input.shape[0] == 0:
        raise ValueError("input must contain at least one token")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")

    device = input.device
    _validate_device(device.index if device.index is not None else torch.cuda.current_device())
    num_tokens = input.shape[0]
    _require_tensor(
        token_selected_experts,
        "token_selected_experts",
        shape=(num_tokens, _TOP_K),
        dtype=torch.int32,
        device=device,
    )
    _require_tensor(
        token_final_scales,
        "token_final_scales",
        shape=(num_tokens, _TOP_K),
        dtype=torch.float32,
        device=device,
    )
    _require_tensor(
        fc1_micro_weights,
        "fc1_micro_weights",
        shape=(_MICRO_SHARDS, _NUM_EXPERTS, _FC1_SIZE, _HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    _require_tensor(
        fc2_micro_weights,
        "fc2_micro_weights",
        shape=(_MICRO_SHARDS, _NUM_EXPERTS, _HIDDEN_SIZE, _INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )

    if output is None:
        return torch.empty_like(input)
    _require_tensor(
        output,
        "output",
        shape=(num_tokens, _HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    return output


def _finalize_with_triton(
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
    grid = (output.shape[0], triton.cdiv(_HIDDEN_SIZE, finalize_block))
    _multi_micro_finalize_kernel[grid](
        permuted_expert_output_0,
        permuted_expert_output_1,
        expanded_to_permuted,
        token_final_scales,
        output,
        hidden_size=_HIDDEN_SIZE,
        TOP_K=_TOP_K,
        BLOCK_SIZE=finalize_block,
        num_warps=4,
    )


def _finalize_with_torch(
    permuted_expert_output_0: torch.Tensor,
    permuted_expert_output_1: torch.Tensor,
    permuted_to_expanded: torch.Tensor,
    token_final_scales: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Correctness fallback for installations without Triton."""

    permuted_expert_output = permuted_expert_output_0.float()
    permuted_expert_output.add_(permuted_expert_output_1.float())
    expanded_output = torch.empty_like(permuted_expert_output)
    expanded_output.index_copy_(0, permuted_to_expanded, permuted_expert_output)
    weighted = expanded_output.view(-1, _TOP_K, _HIDDEN_SIZE)
    weighted.mul_(token_final_scales.unsqueeze(-1))
    output.copy_(weighted.sum(dim=1))


def lightx2v_multi_micro_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_micro_weights: torch.Tensor,
    fc2_micro_weights: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    backend: str = "grouped_mm",
) -> torch.Tensor:
    """Run the HunyuanImage-3 two-micro-shard MoE without weight movement.

    Expert weights retain a leading micro-shard dimension. ``output`` is
    optional; when supplied it is filled in-place and returned.

    Notes
    -----
    Expert ids are a trusted router result and must be in ``[0, 64)``.  The
    range is not checked here because reading a CUDA min/max in Python would
    introduce a device synchronization into every MoE layer.
    """

    output = _validate_inputs(
        input,
        token_selected_experts,
        token_final_scales,
        fc1_micro_weights,
        fc2_micro_weights,
        output,
        backend,
    )

    # Shared routing: sorted route i maps to expanded route
    # (token=i//top_k, slot=i%top_k).  Both micro shards reuse this exact
    # permutation and the same cumulative expert offsets.
    flat_experts = token_selected_experts.reshape(-1)
    permuted_to_expanded = torch.argsort(flat_experts)
    permuted_token_indices = torch.div(permuted_to_expanded, _TOP_K, rounding_mode="floor")
    permuted_input = input.index_select(0, permuted_token_indices)
    expert_offsets = torch.bincount(flat_experts, minlength=_NUM_EXPERTS).cumsum(0).to(torch.int32)

    micro_activations: list[torch.Tensor] = []
    for micro_idx in range(_MICRO_SHARDS):
        projected = torch._grouped_mm(
            permuted_input,
            fc1_micro_weights[micro_idx].transpose(1, 2),
            offs=expert_offsets,
        )
        gate, up = projected.chunk(2, dim=-1)
        # FlashInfer/TensorRT-LLM's default SwiGLU convention for the packed
        # Hunyuan weights is first-half * SiLU(second-half).
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
        _finalize_with_triton(
            permuted_expert_outputs[0],
            permuted_expert_outputs[1],
            permuted_to_expanded,
            token_final_scales,
            output,
        )
    else:  # pragma: no cover - production environment includes Triton.
        _finalize_with_torch(
            permuted_expert_outputs[0],
            permuted_expert_outputs[1],
            permuted_to_expanded,
            token_final_scales,
            output,
        )
    return output
