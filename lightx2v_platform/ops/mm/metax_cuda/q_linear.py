import torch
from torch import nn

from .kernels import metax_int8_linear


class MetaxQuantLinearInt8(nn.Module):
    """Checkpoint-compatible INT8 linear for the T5 and CLIP encoders."""

    def __init__(self, in_features, out_features, bias=True, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty((out_features, 1), dtype=torch.float32))
        self.register_buffer("bias", torch.empty(out_features, dtype=dtype) if bias else None)

    def forward(self, input_tensor):
        return metax_int8_linear(input_tensor, self.weight.t(), self.weight_scale, self.bias)

    def _apply(self, fn, recurse=True):
        # Device/dtype conversions must preserve INT8 weights and FP32 scales.
        quantized_buffers = {id(self.weight), id(self.weight_scale)}

        def preserve_quantized_dtype(tensor):
            transformed = fn(tensor)
            if id(tensor) in quantized_buffers and transformed.dtype != tensor.dtype:
                return tensor.to(device=transformed.device)
            return transformed

        return super()._apply(preserve_quantized_dtype, recurse=recurse)
