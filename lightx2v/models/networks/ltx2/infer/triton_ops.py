from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _fused_rmsnorm_modulate_kernel(
    X,  # Input [M, N]
    Y,  # Output [M, N]
    Scale,  # Scale tensor (various shapes supported)
    Shift,  # Shift tensor (various shapes supported)
    W,  # Optional weight for RMSNorm [N]
    B,  # Optional bias for RMSNorm [N]
    stride_x_row,
    stride_y_row,
    stride_scale_row,  # For 2D/3D scale
    stride_shift_row,  # For 2D/3D shift
    M,
    N,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SCALE_IS_4D: tl.constexpr,  # [B, F, 1, C] format
    num_frames: tl.constexpr,  # For 4D scale/shift
    frame_seqlen: tl.constexpr,  # For 4D scale/shift
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load input
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

    # Step 1: RMSNorm
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    x_hat = x * rstd

    # Apply optional weight and bias for RMSNorm
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        x_hat = x_hat * w

    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x_hat + b

    # Step 2: Load scale and shift based on format
    if SCALE_IS_4D:
        # For 4D: [B, F, 1, C] -> need to map row to correct frame
        batch_idx = row // (num_frames * frame_seqlen)
        t_idx = row % (num_frames * frame_seqlen)
        frame_idx = t_idx // frame_seqlen
        scale_row_idx = batch_idx * num_frames + frame_idx

        Scale += scale_row_idx * N
        Shift += scale_row_idx * N
        scale = tl.load(Scale + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(Shift + cols, mask=mask, other=0.0).to(tl.float32)
    else:
        # For 2D/3D: direct row indexing with stride
        Scale += row * stride_scale_row
        Shift += row * stride_shift_row
        scale = tl.load(Scale + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(Shift + cols, mask=mask, other=0.0).to(tl.float32)

    # Step 3: Apply modulation: x_hat * (1 + scale) + shift
    y = x_hat * (1.0 + scale) + shift

    tl.store(Y + cols, y, mask=mask)


def fused_rmsnorm_modulate(
    x: Tensor,
    scale: Tensor,
    shift: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-6,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    融合的 RMSNorm + Modulation 操作

    计算: (RMSNorm(x) * weight + bias) * (1 + scale) + shift

    Args:
        x: 输入张量 [B, L, C] 或 [M, N]
        scale: 调制缩放张量，支持多种形状:
            - [B, F, 1, C]: 4D 格式（帧级调制）
            - [B, L, C] 或 [B, 1, C]: 3D 格式
            - [1, C]: 2D 格式
        shift: 调制偏移张量，形状同 scale
        weight: RMSNorm 权重 [C]，可选
        bias: RMSNorm 偏置 [C]，可选
        eps: RMSNorm 的 epsilon
        out: 输出张量，可选

    Returns:
        调制后的张量，形状与 x 相同
    """
    # Reshape to 2D for processing
    original_shape = x.shape
    if x.dim() == 3:
        B, L, C = x.shape
        x_2d = x.reshape(B * L, C)
    elif x.dim() == 2:
        x_2d = x
        B, L = 1, x.shape[0]
        C = x.shape[1]
    else:
        raise ValueError(f"Input must be 2D or 3D, got {x.dim()}D")

    M, N = x_2d.shape
    x_2d = x_2d.contiguous()

    # Validate weight and bias
    if weight is not None:
        assert weight.shape == (N,) and weight.stride(-1) == 1
    if bias is not None:
        assert bias.shape == (N,) and bias.stride(-1) == 1

    # Prepare output
    if out is None:
        out_2d = torch.empty_like(x_2d)
    else:
        out_2d = out.reshape(M, N) if out.dim() == 3 else out

    # Determine scale/shift format
    is_4d = scale.dim() == 4

    if is_4d:
        # [B, F, 1, C] format
        assert scale.shape[0] == B or scale.shape[0] == 1
        num_frames = scale.shape[1]
        assert scale.shape[2] == 1
        assert scale.shape[3] == C
        assert L % num_frames == 0, f"seq_len {L} must be divisible by num_frames {num_frames}"
        frame_seqlen = L // num_frames

        # Reshape to [B*F, C] for easier indexing
        scale_2d = scale.squeeze(2).reshape(-1, C).contiguous()
        shift_2d = shift.squeeze(2).reshape(-1, C).contiguous()
        stride_scale_row = 0
        stride_shift_row = 0
    else:
        # Handle 2D/3D formats
        if scale.dim() == 2:
            # [B, C] or [1, C] -> expand to [M, C]
            scale_exp = scale.expand(M, N).contiguous()
            shift_exp = shift.expand(M, N).contiguous()
        elif scale.dim() == 3:
            # [B, L, C] -> reshape to [M, C]
            scale_exp = scale.reshape(M, N).contiguous()
            shift_exp = shift.reshape(M, N).contiguous()
        else:
            raise ValueError(f"scale must be 2D, 3D, or 4D, got {scale.dim()}D")

        scale_2d = scale_exp
        shift_2d = shift_exp
        stride_scale_row = scale_2d.stride(0)
        stride_shift_row = shift_2d.stride(0)
        num_frames = 0
        frame_seqlen = 0

    # Kernel configuration
    MAX_FUSED_SIZE = 65536 // x_2d.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This kernel doesn't support feature dim >= 64KB.")

    num_warps = min(max(BLOCK_N // 256, 1), 8)

    # Launch kernel
    grid = (M,)
    _fused_rmsnorm_modulate_kernel[grid](
        x_2d,
        out_2d,
        scale_2d,
        shift_2d,
        weight if weight is not None else x_2d,  # dummy when HAS_WEIGHT=False
        bias if bias is not None else x_2d,  # dummy when HAS_BIAS=False
        x_2d.stride(0),
        out_2d.stride(0),
        stride_scale_row,
        stride_shift_row,
        M,
        N,
        eps,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        SCALE_IS_4D=is_4d,
        num_frames=num_frames if is_4d else 0,
        frame_seqlen=frame_seqlen if is_4d else 0,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    # Reshape back to original shape
    return out_2d.reshape(original_shape) if out is None else out


# 测试函数
def test_fused_rmsnorm_modulate():
    """测试融合 kernel 的正确性"""
    B, L, C = 2, 128, 1024
    eps = 1e-6

    # 创建测试数据
    x = torch.randn(B, L, C, device="cuda", dtype=torch.float32)
    scale = torch.randn(B, L, C, device="cuda", dtype=torch.float32)
    shift = torch.randn(B, L, C, device="cuda", dtype=torch.float32)
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)

    # Torch 参考实现（不带 weight/bias）
    def reference_impl(x, scale, shift, eps):
        return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)) * (1 + scale) + shift

    # 测试不带 weight/bias
    out_ref = reference_impl(x, scale, shift, eps)
    out_triton = fused_rmsnorm_modulate(x, scale, shift, eps=eps)

    print("测试不带 weight/bias:")
    print(f"  最大误差: {(out_ref - out_triton).abs().max().item():.6e}")
    print(f"  平均误差: {(out_ref - out_triton).abs().mean().item():.6e}")
    assert torch.allclose(out_ref, out_triton, rtol=1e-4, atol=1e-5), "不带 weight/bias 的测试失败"

    # 测试带 weight/bias
    def reference_impl_with_wb(x, scale, shift, weight, bias, eps):
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = normed * weight + bias
        return normed * (1 + scale) + shift

    out_ref_wb = reference_impl_with_wb(x, scale, shift, weight, bias, eps)
    out_triton_wb = fused_rmsnorm_modulate(x, scale, shift, weight, bias, eps=eps)

    print("\n测试带 weight/bias:")
    print(f"  最大误差: {(out_ref_wb - out_triton_wb).abs().max().item():.6e}")
    print(f"  平均误差: {(out_ref_wb - out_triton_wb).abs().mean().item():.6e}")
    assert torch.allclose(out_ref_wb, out_triton_wb, rtol=1e-4, atol=1e-5), "带 weight/bias 的测试失败"

    # 测试 4D scale/shift 格式
    num_frames = 8
    scale_4d = torch.randn(B, num_frames, 1, C, device="cuda", dtype=torch.float32)
    shift_4d = torch.randn(B, num_frames, 1, C, device="cuda", dtype=torch.float32)

    # 扩展 scale_4d 到 [B, L, C] 用于参考实现
    frame_len = L // num_frames
    scale_expanded = scale_4d.squeeze(2).repeat_interleave(frame_len, dim=1)
    shift_expanded = shift_4d.squeeze(2).repeat_interleave(frame_len, dim=1)

    out_ref_4d = reference_impl(x, scale_expanded, shift_expanded, eps)
    out_triton_4d = fused_rmsnorm_modulate(x, scale_4d, shift_4d, eps=eps)

    print("\n测试 4D scale/shift 格式:")
    print(f"  最大误差: {(out_ref_4d - out_triton_4d).abs().max().item():.6e}")
    print(f"  平均误差: {(out_ref_4d - out_triton_4d).abs().mean().item():.6e}")
    assert torch.allclose(out_ref_4d, out_triton_4d, rtol=1e-4, atol=1e-5), "4D 格式测试失败"

    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    test_fused_rmsnorm_modulate()
