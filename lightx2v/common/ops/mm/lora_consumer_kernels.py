"""LoRA consumer kernels with explicit eager dtype/rounding boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu(BASE, DELTA, OUT, N: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < TOTAL
    source = offset // N * (2 * N) + offset % N
    dtype = OUT.dtype.element_ty
    value = (tl.load(BASE + source, valid, 0).to(tl.float32) + tl.load(DELTA + source, valid, 0).to(tl.float32)).to(dtype).to(tl.float32)
    gate = (tl.load(BASE + source + N, valid, 0).to(tl.float32) + tl.load(DELTA + source + N, valid, 0).to(tl.float32)).to(dtype).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(dtype).to(tl.float32)
    tl.store(OUT + offset, value * activated, valid)


@triton.jit
def _gated_residual(
    BASE, DELTA, RESIDUAL, GATE, INDICES, OUT, N: tl.constexpr, GATE_STRIDE: tl.constexpr, TOTAL: tl.constexpr, SUM_DTYPE: tl.constexpr, PRODUCT_DTYPE: tl.constexpr, BLOCK: tl.constexpr
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < TOTAL
    row = tl.load(INDICES + offset // N, valid, 0).to(tl.int64)
    gate = tl.load(GATE + row * GATE_STRIDE + offset % N, valid, 0).to(tl.float32)
    projected = (tl.load(BASE + offset, valid, 0).to(tl.float32) + tl.load(DELTA + offset, valid, 0).to(tl.float32)).to(SUM_DTYPE).to(tl.float32)
    gated = (gate * projected).to(PRODUCT_DTYPE).to(tl.float32)
    residual = tl.load(RESIDUAL + offset, valid, 0).to(tl.float32)
    tl.store(OUT + offset, residual + gated, valid)


_DTYPES = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


# Keep the explicit rounding and no-FMA contract opaque to Inductor.
@torch.library.custom_op("lightx2v::lora_consumer_swiglu", mutates_args=(), device_types="cuda")
def swiglu(base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    dtype = base.dtype
    output = torch.empty((base.shape[0], base.shape[1] // 2), dtype=dtype, device=base.device)
    _swiglu[(triton.cdiv(output.numel(), 256),)](base.contiguous(), delta.contiguous(), output, output.shape[1], output.numel(), 256, enable_fp_fusion=False)
    return output


@swiglu.register_fake
def _swiglu_fake(base, delta):
    return base.new_empty((base.shape[0], base.shape[1] // 2))


@torch.library.custom_op("lightx2v::lora_consumer_gated_residual", mutates_args=(), device_types="cuda")
def gated_residual(base: torch.Tensor, delta: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    sum_dtype = base.dtype
    product_dtype = torch.promote_types(sum_dtype, gate.dtype)
    output = torch.empty(residual.shape, dtype=torch.promote_types(product_dtype, residual.dtype), device=residual.device)
    gate = gate.contiguous()
    _gated_residual[(triton.cdiv(output.numel(), 256),)](
        base.contiguous(),
        delta.contiguous(),
        residual.contiguous(),
        gate,
        indices.contiguous(),
        output,
        output.shape[1],
        gate.stride(0),
        output.numel(),
        _DTYPES[sum_dtype],
        _DTYPES[product_dtype],
        256,
        enable_fp_fusion=False,
    )
    return output


@gated_residual.register_fake
def _gated_residual_fake(base, delta, residual, gate, indices):
    dtype = torch.promote_types(base.dtype, gate.dtype)
    return residual.new_empty(residual.shape, dtype=torch.promote_types(dtype, residual.dtype))
