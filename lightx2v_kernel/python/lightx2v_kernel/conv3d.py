import torch

def _empty_fp8_conv3d_output(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    output_frames = (input_tensor.size(2) - weight.size(2)) // stride_d + 1
    output_height = (input_tensor.size(3) - weight.size(3)) // stride_h + 1
    output_width = (input_tensor.size(4) - weight.size(4)) // stride_w + 1
    return torch.empty(
        (
            input_tensor.size(0),
            weight.size(0),
            output_frames,
            output_height,
            output_width,
        ),
        device=input_tensor.device,
        dtype=dtype,
        memory_format=torch.channels_last_3d,
    )


@torch.library.register_fake("lightx2v_kernel::fp8_conv3d_f16_accum_sm120")
def _fp8_conv3d_f16_accum_sm120_fake(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride_d: int,
    stride_h: int,
    stride_w: int,
) -> torch.Tensor:
    return _empty_fp8_conv3d_output(
        input_tensor, weight, stride_d, stride_h, stride_w, torch.float16
    )


@torch.library.register_fake("lightx2v_kernel::fp8_conv3d_f32_accum_sm120")
def _fp8_conv3d_f32_accum_sm120_fake(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride_d: int,
    stride_h: int,
    stride_w: int,
) -> torch.Tensor:
    return _empty_fp8_conv3d_output(
        input_tensor, weight, stride_d, stride_h, stride_w, torch.float32
    )


def _fp8_conv3d_f16_accum_sm120(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int, int],
) -> torch.Tensor:
    return torch.ops.lightx2v_kernel.fp8_conv3d_f16_accum_sm120.default(
        input_tensor,
        weight,
        stride[0],
        stride[1],
        stride[2],
    )


def _fp8_conv3d_f32_accum_sm120(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int, int],
) -> torch.Tensor:
    return torch.ops.lightx2v_kernel.fp8_conv3d_f32_accum_sm120.default(
        input_tensor,
        weight,
        stride[0],
        stride[1],
        stride[2],
    )


def fp8_conv3d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    stride: tuple[int, int, int],
    accumulator_dtype: torch.dtype,
) -> torch.Tensor:
    """Run valid E4M3 Conv3D on channels-last-3d SM120 tensors.

    Padding, bias, dilation and grouped convolution are not supported.
    """

    if input_tensor.device.type != "cuda":
        raise RuntimeError(f"FP8 Conv3D requires a CUDA input, got {input_tensor.device}")
    capability = torch.cuda.get_device_capability(input_tensor.device)
    if capability != (12, 0):
        raise RuntimeError(f"No FP8 Conv3D backend is available for SM{capability[0]}{capability[1]}")
    if accumulator_dtype == torch.float16:
        return _fp8_conv3d_f16_accum_sm120(input_tensor, weight, stride)
    if accumulator_dtype == torch.float32:
        return _fp8_conv3d_f32_accum_sm120(input_tensor, weight, stride)
    raise ValueError(f"FP8 Conv3D does not support {accumulator_dtype} accumulation")
