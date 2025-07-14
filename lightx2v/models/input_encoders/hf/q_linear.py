import torch
import torch.nn as nn
from vllm import _custom_ops as ops


class QuantLinearInt8(nn.Module):
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
        input_tensor_quant, input_tensor_scale, _ = ops.scaled_int8_quant(x, scale=None, azp=None, symmetric=True)
        return input_tensor_quant, input_tensor_scale

    def forward(self, input_tensor):
        input_tensor = input_tensor.squeeze(0)
        shape = (input_tensor.shape[0], self.weight.shape[0])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)

        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        torch.ops._C.cutlass_scaled_mm(
            output_tensor,
            input_tensor_quant,
            self.weight.t(),
            input_tensor_scale,
            self.weight_scale.float(),
            self.bias,
        )
        return output_tensor.unsqueeze(0)

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


class QuantLinearFp8(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn))
        self.register_buffer("weight_scale", torch.empty((out_features, 1), dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=dtype))
        else:
            self.register_buffer("bias", None)

    def act_quant_func(self, x):
        input_tensor_quant, input_tensor_scale = ops.scaled_fp8_quant(x, None, scale_ub=None, use_per_token_if_dynamic=True)
        return input_tensor_quant, input_tensor_scale

    def forward(self, input_tensor):
        input_tensor = input_tensor.squeeze(0)
        shape = (input_tensor.shape[0], self.weight.shape[0])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        torch.ops._C.cutlass_scaled_mm(
            output_tensor,
            input_tensor_quant,
            self.weight.t(),
            input_tensor_scale,
            self.weight_scale.float(),
            self.bias,
        )

        return output_tensor.unsqueeze(0)

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
