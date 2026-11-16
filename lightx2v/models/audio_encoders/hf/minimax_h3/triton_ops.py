"""FP32 anti-aliased SnakeBeta for H3's 2x, 12-tap audio decoder."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _upsample_snake_kernel(x, up_filter, alpha, beta, output, length, channels: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(1)
    t = tl.program_id(0) * block + tl.arange(0, block)
    value = tl.full((block,), 0, tl.float32)
    # After input padding and output crop, transpose-conv writes t = 2*i + k - 5.
    for k in tl.static_range(12):
        pos = t + 5 - k
        index = tl.minimum(tl.maximum(pos // 2, 0), length - 1)
        sample = tl.load(x + row * length + index, mask=(t < 2 * length) & ((pos & 1) == 0), other=0)
        value += sample * tl.load(up_filter + k)
    value *= 2.0
    alpha_value = libdevice.exp(tl.load(alpha + row % channels))
    beta_value = libdevice.exp(tl.load(beta + row % channels))
    sine = libdevice.sin(alpha_value * value)
    value += libdevice.rcp_rn(beta_value + 1.0e-9) * (sine * sine)
    tl.store(output + row * (2 * length) + t, value, mask=t < 2 * length)


@triton.jit
def _downsample_kernel(x, down_filter, output, length, block: tl.constexpr):
    row = tl.program_id(1)
    t = tl.program_id(0) * block + tl.arange(0, block)
    value = tl.full((block,), 0, tl.float32)
    for k in tl.static_range(12):
        # Replicate the cropped upsampled signal's endpoints before filtering.
        index = tl.minimum(tl.maximum(2 * t - 5 + k, 0), 2 * length - 1)
        sample = tl.load(x + row * (2 * length) + index, mask=t < length, other=0)
        value += sample * tl.load(down_filter + k)
    tl.store(output + row * length + t, value, mask=t < length)


def anti_alias_activation(hidden_states, up_filter, down_filter, alpha, beta):
    """Apply anti-aliased SnakeBeta to a contiguous FP32 CUDA activation.

    The H3 module supplies fixed 12-tap filters and log-scale alpha/beta.
    Direct launches use separate multiply/add rounding in the audio decode graphs.
    """
    if hidden_states.dtype != torch.float32 or not hidden_states.is_contiguous():
        raise ValueError("H3 audio activation requires a contiguous FP32 input")
    batch, channels, length = hidden_states.shape
    upsampled = hidden_states.new_empty((batch, channels, 2 * length))
    output = torch.empty_like(hidden_states)
    with torch.cuda.device(hidden_states.device):
        _upsample_snake_kernel[(triton.cdiv(2 * length, 256), batch * channels)](hidden_states, up_filter, alpha, beta, upsampled, length, channels, 256, enable_fp_fusion=False)
        _downsample_kernel[(triton.cdiv(length, 256), batch * channels)](upsampled, down_filter, output, length, 256, enable_fp_fusion=False)
    return output
