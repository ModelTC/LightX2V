import torch
import triton
import triton.language as tl

from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


@triton.jit
def _residual_norm_scale_kernel(X, Update, Gate, Scale, Residual, Hidden, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    offsets = row * WIDTH + cols
    dtype = X.dtype.element_ty
    x = tl.load(X + offsets, mask, other=0).to(tl.float32)
    update = tl.load(Update + offsets, mask, other=0).to(tl.float32)
    gate = tl.load(Gate + cols, mask, other=0).to(tl.float32)
    scale = tl.load(Scale + cols, mask, other=0).to(tl.float32)

    # Preserve the eager dtype round-trips before addition and normalization.
    product = (update * gate).to(dtype).to(tl.float32)
    residual = (x + product).to(dtype).to(tl.float32)
    tl.store(Residual + offsets, residual, mask)
    mean = tl.sum(residual, 0) / WIDTH
    centered = tl.where(mask, residual - mean, 0.0)
    variance = tl.sum(centered * centered, 0) / WIDTH
    normalized = (centered * tl.rsqrt(variance + EPS)).to(dtype).to(tl.float32)
    tl.store(Hidden + offsets, normalized * scale, mask)


def fused_residual_norm_scale(x, update, gate, scale, eps):
    """Return x + gate * update and LayerNorm(residual) * scale for contiguous [L, D] inputs."""
    residual = torch.empty_like(x)
    hidden = torch.empty_like(x)
    width = x.shape[-1]
    with torch_device_module.device(x.device):
        _residual_norm_scale_kernel[(x.shape[0],)](x, update, gate, scale, residual, hidden, width, eps, triton.next_power_of_2(width), enable_fp_fusion=False)
    return residual, hidden


@triton.jit
def _silu_mul_kernel(Gate, Up, Output, NUMEL: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUMEL
    gate = tl.load(Gate + offsets, mask, other=0).to(tl.float32)
    up = tl.load(Up + offsets, mask, other=0).to(tl.float32)
    # F.silu returns the input dtype before the separate multiply in eager mode.
    activated = tl.div_rn(gate, 1.0 + tl.exp(-gate)).to(Gate.dtype.element_ty).to(tl.float32)
    tl.store(Output + offsets, activated * up, mask)


def fused_silu_mul(gate, up):
    """Compute silu(gate) * up for equally shaped contiguous projection outputs."""
    output = torch.empty_like(gate)
    with torch_device_module.device(gate.device):
        _silu_mul_kernel[(triton.cdiv(gate.numel(), 1024),)](gate, up, output, gate.numel(), 1024, enable_fp_fusion=False)
    return output


@triton.jit
def _residual_add_kernel(X, Update, Gate, Output, NUMEL: tl.constexpr, WIDTH: tl.constexpr, CLAMP: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUMEL
    dtype = X.dtype.element_ty
    x = tl.load(X + offsets, mask, other=0).to(tl.float32)
    update = tl.load(Update + offsets, mask, other=0).to(tl.float32)
    gate = tl.load(Gate + offsets % WIDTH, mask, other=0).to(tl.float32)
    product = (update * gate).to(dtype).to(tl.float32)
    result = (x + product).to(dtype).to(tl.float32)
    if CLAMP:
        # Comparisons preserve NaNs, matching torch.clamp.
        result = tl.where(result > 65504.0, 65504.0, result)
        result = tl.where(result < -65504.0, -65504.0, result)
    tl.store(Output + offsets, result, mask)


def fused_residual_add(x, update, gate):
    """Compute x + gate * update, including the block's final FP16 clamp."""
    output = torch.empty_like(x)
    with torch_device_module.device(x.device):
        _residual_add_kernel[(triton.cdiv(x.numel(), 1024),)](x, update, gate, output, x.numel(), x.shape[-1], x.dtype == torch.float16, 1024, enable_fp_fusion=False)
    return output
