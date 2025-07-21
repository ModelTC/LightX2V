import torch


def cutlass_scaled_nvfp4_mm(mat_a, mat_b, scales_a, scales_b, alpha, bias=None):
    m, n = mat_a.shape[0], mat_b.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=mat_a.device)
    torch.ops.lightx2v_kernel.cutlass_scaled_nvfp4_mm_sm120.default(out, mat_a, mat_b, scales_a, scales_b, alpha, bias)
    return out


def scaled_nvfp4_quant(input: torch.Tensor, input_global_scale: torch.Tensor):
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in a sizzled layout.
    """
    # assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    # other_dims = 1 if input.ndim == 1 else -1
    # input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    # assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    # assert input.dtype in (
    #     torch.float16,
    #     torch.bfloat16,
    # ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Then, the scaling
    # factors in float8_e4m3fn are packed into an int32 for every 4 values.
    # rounded_m = ((m + 128 - 1) // 128) * 128
    # scale_n = n // block_size
    # rounded_n = ((scale_n + 4 - 1) // 4) * 4
    output_scale = torch.zeros((((m + 128 - 1) // 128) * 128, (n // block_size + 4 - 1) // 4), device=device, dtype=torch.int32)

    torch.ops.lightx2v_kernel.scaled_nvfp4_quant_sm120.default(output, input, output_scale, input_global_scale)
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


def scaled_mxfp4_quant(input: torch.Tensor):
    m, n = input.shape
    block_size = 32
    device = input.device

    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)
    output_scale = torch.zeros(((m + 128 - 1) // 128 * 128, (n // block_size + 4 - 1) // 4), device=device, dtype=torch.int32)

    torch.ops.lightx2v_kernel.scaled_mxfp4_quant_sm120.default(output, input, output_scale)
    output_scale = output_scale.view(torch.float8_e8m0fnu)
    return output, output_scale


def scaled_mxfp6_quant(input: torch.Tensor):
    m, n = input.shape
    block_size = 32
    device = input.device

    output = torch.empty((m, 3 * n // 4), device=device, dtype=torch.uint8)
    output_scale = torch.zeros(((m + 128 - 1) // 128 * 128, (n // block_size + 4 - 1) // 4), device=device, dtype=torch.int32)

    torch.ops.lightx2v_kernel.scaled_mxfp6_quant_sm120.default(output, input, output_scale)
    output_scale = output_scale.view(torch.float8_e8m0fnu)
    return output, output_scale


def scaled_mxfp8_quant(input: torch.Tensor):
    m, n = input.shape
    block_size = 32
    device = input.device

    output = torch.empty((m, n), device=device, dtype=torch.uint8)
    output_scale = torch.empty(((m + 128 - 1) // 128 * 128, (n // block_size + 4 - 1) // 4), device=device, dtype=torch.int32)

    torch.ops.lightx2v_kernel.scaled_mxfp8_quant_sm120.default(output, input, output_scale)
    output_scale = output_scale.view(torch.float8_e8m0fnu)
    return output, output_scale


def cutlass_scaled_mxfp4_mm(mat_a, mat_b, scales_a, scales_b, alpha, bias=None):
    m, n = mat_a.shape[0], mat_b.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=mat_a.device)
    torch.ops.lightx2v_kernel.cutlass_scaled_mxfp4_mm_sm120.default(out, mat_a, mat_b, scales_a, scales_b, alpha, bias)
    return out


def cutlass_scaled_mxfp6_mxfp8_mm(mat_a, mat_b, scales_a, scales_b, alpha, bias=None):
    m, n = mat_a.shape[0], mat_b.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=mat_a.device)
    torch.ops.lightx2v_kernel.cutlass_scaled_mxfp6_mxfp8_mm_sm120.default(out, mat_a, mat_b, scales_a, scales_b, alpha, bias)
    return out


def cutlass_scaled_mxfp8_mm(mat_a, mat_b, scales_a, scales_b, alpha, bias=None):
    m, n = mat_a.shape[0], mat_b.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=mat_a.device)
    torch.ops.lightx2v_kernel.cutlass_scaled_mxfp8_mm_sm120.default(out, mat_a, mat_b, scales_a, scales_b, alpha, bias)
    return out
