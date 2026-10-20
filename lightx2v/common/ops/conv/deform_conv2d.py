import torch
import torch.nn.functional as F

from lightx2v.utils.registry_factory import CONV2D_WEIGHT_REGISTER, MM_WEIGHT_REGISTER

from .conv2d import Conv2dWeightTemplate


def deformable_columns(input_tensor, offset, mask, kernel_size, stride, padding, dilation, start, stop):
    """Bilinearly sample one output tile, in Conv2d's [channel, ky, kx] order."""
    batch, channels, height, width = input_tensor.shape
    kh, kw = kernel_size
    out_h, out_w = offset.shape[-2:]
    points = torch.arange(start, stop, device=input_tensor.device)
    ky, kx = torch.meshgrid(torch.arange(kh, device=input_tensor.device), torch.arange(kw, device=input_tensor.device), indexing="ij")
    shifts = offset.reshape(batch, kh * kw, 2, out_h * out_w)[..., start:stop].float()
    y = (points // out_w * stride[0] - padding[0])[None, None] + ky.flatten()[None, :, None] * dilation[0] + shifts[:, :, 0]
    x = (points % out_w * stride[1] - padding[1])[None, None] + kx.flatten()[None, :, None] * dilation[1] + shifts[:, :, 1]
    grid = torch.stack(((x + 0.5) * (2 / width) - 1, (y + 0.5) * (2 / height) - 1), dim=-1)
    sampled = F.grid_sample(input_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if mask is not None:
        sampled = sampled * mask.flatten(2)[:, None, :, start:stop].float()
    return sampled.permute(0, 3, 1, 2).reshape(batch * (stop - start), channels * kh * kw)


@CONV2D_WEIGHT_REGISTER("torch_deform_bf16")
class DeformConv2dWeight(Conv2dWeightTemplate):
    """FP32 interpolation followed by a registered BF16 projection GEMM."""

    def __init__(self, weight_name, bias_name, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, linear_backend="Default", tile_size=4096):
        super().__init__(weight_name, bias_name, stride, padding, dilation, groups)
        if groups != 1:
            raise ValueError("torch_deform_bf16 supports one weight group and one offset group")
        self.projection = MM_WEIGHT_REGISTER[linear_backend](weight_name, bias_name)
        self.tile_size = tile_size

    def load(self, weight_dict):
        weight = weight_dict[self.weight_name]
        self.kernel_size = tuple(weight.shape[-2:])
        self.out_channels = weight.shape[0]
        tensors = {self.weight_name: weight.flatten(1)}
        if self.bias_name is not None:
            tensors[self.bias_name] = weight_dict[self.bias_name]
        self.projection.load(tensors)
        if self.projection.weight.dtype != torch.bfloat16:
            raise ValueError("Deformable convolution requires BF16 projection weights")

    def apply(self, input_tensor, offset, mask=None):
        batch = input_tensor.shape[0]
        out_h, out_w = offset.shape[-2:]
        output = input_tensor.new_empty((batch, self.out_channels, out_h * out_w), dtype=torch.bfloat16)
        # Interpolation is coordinate math; all projection operands are BF16.
        with torch.autocast(input_tensor.device.type, enabled=False):
            values = input_tensor.float()
            for start in range(0, out_h * out_w, self.tile_size):
                stop = min(start + self.tile_size, out_h * out_w)
                columns = deformable_columns(values, offset, mask, self.kernel_size, self.stride, self.padding, self.dilation, start, stop)
                projected = self.projection.apply(columns.to(torch.bfloat16).contiguous())
                output[:, :, start:stop] = projected.reshape(batch, stop - start, self.out_channels).transpose(1, 2)
        return output.reshape(batch, self.out_channels, out_h, out_w)
