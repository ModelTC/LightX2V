import torch

try:
    import sgl_kernel
except ImportError:
    sgl_kernel = None


def sgl_fp8_scaled_mm_meta(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty((mat_a.shape[0], mat_b.shape[1]), dtype=out_dtype, device=mat_a.device)


if sgl_kernel is not None and hasattr(torch.ops.sgl_kernel, "fp8_scaled_mm"):
    _sgl_fp8_scaled_mm_op = torch.ops.sgl_kernel.fp8_scaled_mm.default

    if not _sgl_fp8_scaled_mm_op.has_kernel_for_dispatch_key("Meta"):
        torch.library.register_fake(_sgl_fp8_scaled_mm_op, sgl_fp8_scaled_mm_meta)


def sgl_fp8_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if sgl_kernel is None:
        raise ImportError("sgl_kernel is required for SGL FP8 scaled matrix multiplication")
    return sgl_kernel.fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=bias)
