import torch
import torch.nn as nn

try:
    from vllm import _custom_ops as ops
except ImportError:
    ops = None

try:
    import sgl_kernel
except ImportError:
    sgl_kernel = None

try:
    from torchao.quantization.utils import quant_int8_per_token_matmul, quantize_activation_per_token_absmax
except ImportError:
    quant_int8_per_token_matmul, quantize_activation_per_token_absmax = None, None

try:
    from q8_kernels.functional.linear import q8_linear
except ImportError:
    q8_linear = None

try:
    from q8_kernels.functional.linear import fp8_linear
except ImportError:
    fp8_linear = None

try:
    import torch_mlu_ops as tmo
except ImportError:
    tmo = None


class VllmQuantLinearInt8(nn.Module):
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


class VllmQuantLinearFp8(nn.Module):
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


class SglQuantLinearFp8(nn.Module):
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
        m, k = x.shape
        input_tensor_quant = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda", requires_grad=False)
        input_tensor_scale = torch.empty((m, 1), dtype=torch.float32, device="cuda", requires_grad=False)
        sgl_kernel.sgl_per_token_quant_fp8(x, input_tensor_quant, input_tensor_scale)
        return input_tensor_quant, input_tensor_scale

    def forward(self, input_tensor):
        input_tensor = input_tensor.squeeze(0)
        shape = (input_tensor.shape[0], self.weight.shape[0])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = sgl_kernel.fp8_scaled_mm(
            input_tensor_quant,
            self.weight.t(),
            input_tensor_scale,
            self.weight_scale,
            dtype,
            bias=self.bias,
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


class TorchaoQuantLinearInt8(nn.Module):
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
        input_tensor_quant, input_tensor_scale = quantize_activation_per_token_absmax(x)
        return input_tensor_quant, input_tensor_scale

    def forward(self, input_tensor):
        input_tensor = input_tensor.squeeze(0)
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = quant_int8_per_token_matmul(input_tensor_quant, input_tensor_scale, self.weight.t(), self.weight_scale.t().float(), output_dtype=torch.bfloat16)
        if self.bias is not None:
            output_tensor = output_tensor + self.bias

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


class Q8FQuantLinearInt8(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty((out_features, 1), dtype=torch.float32))

        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)

    def act_quant_func(self, x):
        input_tensor_quant, input_tensor_scale, _ = ops.scaled_int8_quant(x, scale=None, azp=None, symmetric=True)
        return input_tensor_quant, input_tensor_scale

    def forward(self, x):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(x)
        output_tensor = q8_linear(
            input_tensor_quant,
            self.weight,
            self.bias if self.bias is not None else None,
            input_tensor_scale,
            self.weight_scale.float(),
            fuse_gelu=False,
            out_dtype=torch.bfloat16,
        )
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


class Q8FQuantLinearFp8(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn))
        self.register_buffer("weight_scale", torch.empty((out_features, 1), dtype=torch.float32))

        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)

    def act_quant_func(self, x):
        input_tensor_quant, input_tensor_scale = ops.scaled_fp8_quant(x.squeeze(0), None, scale_ub=None, use_per_token_if_dynamic=True)
        return input_tensor_quant, input_tensor_scale

    def forward(self, x):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(x)
        output_tensor = fp8_linear(
            input_tensor_quant,
            self.weight,
            self.bias if self.bias is not None else None,
            input_tensor_scale,
            self.weight_scale,
            out_dtype=torch.bfloat16,
        )
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


class MluQuantLinearInt8(VllmQuantLinearInt8):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias, dtype)

    def act_quant_func(self, x):
        input_tensor_quant, input_tensor_scale = tmo.scaled_quantize(x)
        return input_tensor_quant, input_tensor_scale

    def forward(self, input_tensor):
        input_tensor = input_tensor.squeeze(0)
        dtype = input_tensor.dtype
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = tmo.scaled_matmul(input_tensor_quant, self.weight, input_tensor_scale, self.weight_scale.squeeze(-1), output_dtype=dtype)
        return output_tensor.unsqueeze(0)