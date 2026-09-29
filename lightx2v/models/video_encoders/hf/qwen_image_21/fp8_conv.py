"""Mixed decoder Conv2D FP8 policy, using H3's Conv3D backend with singleton depth."""

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8_DECODER_PROFILE_KEY = "qwen_image_21_vae_decoder_fp8_profile"
FP8_DECODER_PROFILE = "mixed_f16_qmax21"
FP8_DECODER_QMAX = 21.0


def is_fp8_decoder_conv(name, shape):
    if not name.startswith("decoder.") or len(shape) != 4:
        return False
    name = name.removeprefix("decoder.")
    # Keep attention, boundary convolutions and unused temporal weights intact.
    if name in ("conv_in", "conv_out") or name.endswith("time_conv") or ".attentions." in name:
        return False
    # The legacy CUTLASS backend requires each activation/output buffer < 2 GiB.
    # These high-resolution layers exceed that limit when decoding a 2K image.
    if name.startswith("up_blocks.4.") or name in ("up_blocks.2.upsampler.resample.1", "up_blocks.3.upsampler.resample.1"):
        return False
    return shape[0] % 16 == 0 and shape[1] % 16 == 0


class FP8Conv2d(nn.Conv2d):
    def _conv_forward(self, x, weight, bias):
        if weight.dtype != torch.float8_e4m3fn:
            return super()._conv_forward(x, weight, bias)
        from lightx2v_kernel.conv3d import fp8_conv3d

        if any(self.padding):
            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        n, _, h, w = x.shape
        output_elements = n * self.out_channels * ((h - self.kernel_size[0]) // self.stride[0] + 1) * ((w - self.kernel_size[1]) // self.stride[1] + 1)
        if max(x.numel(), weight.numel(), output_elements * 2) >= 2**31:
            raise ValueError("FP8 VAE convolution exceeds the CUTLASS 2 GiB buffer limit; use the BF16 VAE for this shape")
        scale = (x.abs().amax().float() / FP8_DECODER_QMAX).clamp_min(torch.finfo(torch.float32).tiny)
        quantized = (x.float() / scale).nan_to_num().clamp(-FP8_DECODER_QMAX, FP8_DECODER_QMAX).to(torch.float8_e4m3fn)
        quantized = quantized.unsqueeze(2).contiguous(memory_format=torch.channels_last_3d)
        output = fp8_conv3d(quantized, weight.unsqueeze(2), (1, *self.stride), torch.float16)
        output = output.float() * scale * self.weight_scale
        if bias is not None:
            output = output + bias.float().view(1, -1, 1, 1, 1)
        return output.squeeze(2).to(x.dtype)
