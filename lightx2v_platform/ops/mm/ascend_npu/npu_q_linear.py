import torch
import torch.nn as nn

try:
    import torch_npu
except ImportError:
    tmo = None


class NpuQuantLinearInt8(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty((out_features, 1), dtype=torch.float32))

        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=dtype))
        else:
            self.register_buffer("bias", None)

    def act_quant_func(self, x):
        input_tensor_quant, input_tensor_scale = torch_npu.npu_dynamic_quant(x)
        return input_tensor_quant, input_tensor_scale

    def forward(self, input_tensor):
        dtype = input_tensor.dtype
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = torch_npu.npu_quant_matmul(input_tensor_quant, self.weight.t(), self.weight_scale.reshape(-1), offset=None, bias=self.bias, pertoken_scale=input_tensor_scale.reshape(-1), output_dtype=dtype)
        if len(output_tensor.shape) == 2:
            return output_tensor.unsqueeze(0)
        elif len(output_tensor.shape) == 3:
            return output_tensor

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        def maybe_cast(t):
            if t is not None and t.device != fn(t).device:
                return fn(t)
            return t

        self.weight = maybe_cast(self.weight)
        self.weight_scale = maybe_cast(self.weight_scale)
        self.bias = maybe_cast(self.bias)
        return self
