"""Share the image input cast across BF16 base and LoRA Q/K/V projections."""

import torch
from peft.tuners.lora.layer import Linear as LoraLinear


class _BranchCast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shared):
        ctx.input_dtype = x.dtype
        return shared

    @staticmethod
    def backward(ctx, grad):
        # Eager promotes each branch's dX to FP32 before accumulating the six gradients.
        return grad.to(ctx.input_dtype), None


def _supports_shared_input(projection):
    if (
        type(projection) is not LoraLinear
        or type(projection.base_layer) is not torch.nn.Linear
        or projection.disable_adapters
        or projection.merged
        or len(projection.active_adapters) != 1
        or not getattr(projection, "cast_input_dtype_enabled", False)
        or projection._forward_pre_hooks
        or projection._forward_hooks
        or projection._backward_hooks
        or projection._backward_pre_hooks
    ):
        return False
    (name,) = projection.active_adapters
    if name not in projection.lora_A or getattr(projection, "lora_variant", None) is None or name in projection.lora_variant:
        return False
    dropout = projection.lora_dropout[name]
    return (type(dropout) is torch.nn.Identity or (type(dropout) is torch.nn.Dropout and dropout.p == 0)) and all(
        type(layer) is torch.nn.Linear and layer.weight.dtype == torch.bfloat16 for layer in (projection.base_layer, projection.lora_A[name], projection.lora_B[name])
    )


def project_image_qkv(x, projections, share_input):
    # Autocast caches casts of trainable FP32 leaves with different gradient semantics.
    if not share_input or x.dtype != torch.float32 or (x.is_leaf and x.requires_grad) or not all(_supports_shared_input(p) for p in projections):
        return tuple(projection(x) for projection in projections)

    shared = x.detach().to(torch.bfloat16)
    outputs = []
    for projection in projections:
        (name,) = projection.active_adapters
        base = projection.base_layer(_BranchCast.apply(x, shared))
        adapter_input = projection.lora_dropout[name](_BranchCast.apply(x, shared))
        adapter = projection.lora_B[name](projection.lora_A[name](adapter_input))
        scale = projection.scaling[name]
        outputs.append(base + (adapter if scale == 1.0 else adapter * scale))
    return tuple(outputs)
