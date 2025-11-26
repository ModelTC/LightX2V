import re
from abc import ABCMeta, abstractmethod

import torch

from lightx2v.utils.envs import *
from lightx2v.utils.ggml_tensor import GGMLTensor
from lightx2v.utils.ggml_tensor import dequantize_tensor as gguf_dequantize_tensor
from lightx2v.utils.global_paras import CALIB
from lightx2v.utils.quant_utils import FloatQuantizer, IntegerQuantizer
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER

try:
    from lightx2v_kernel.gemm import (
        cutlass_scaled_mxfp4_mm,
        cutlass_scaled_mxfp6_mxfp8_mm,
        cutlass_scaled_mxfp8_mm,
        cutlass_scaled_nvfp4_mm,
        scaled_mxfp4_quant,
        scaled_mxfp6_quant,
        scaled_mxfp8_quant,
        scaled_nvfp4_quant,
    )
except ImportError:
    scaled_nvfp4_quant, cutlass_scaled_nvfp4_mm = None, None
    scaled_mxfp4_quant, cutlass_scaled_mxfp4_mm = None, None
    scaled_mxfp6_quant, cutlass_scaled_mxfp6_mxfp8_mm = None, None
    scaled_mxfp8_quant, cutlass_scaled_mxfp8_mm = None, None

try:
    from vllm import _custom_ops as ops
except ImportError:
    ops = None

try:
    import sgl_kernel
except ImportError:
    sgl_kernel = None

try:
    from q8_kernels.functional.linear import q8_linear
except ImportError:
    q8_linear = None

try:
    from q8_kernels.functional.linear import fp8_linear
except ImportError:
    fp8_linear = None

try:
    import deep_gemm
except ImportError:
    deep_gemm = None

try:
    from torchao.quantization.utils import quant_int8_per_token_matmul, quantize_activation_per_token_absmax
except ImportError:
    quant_int8_per_token_matmul, quantize_activation_per_token_absmax = None, None

try:
    import gguf
except ImportError:
    gguf = None

try:
    import marlin_cuda_quant
except ImportError:
    marlin_cuda_quant = None

try:
    import torch_mlu_ops as tmo
except ImportError:
    tmo = None


class MMWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.create_cuda_buffer = create_cuda_buffer
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.is_post_adapter = is_post_adapter
        self.config = {}

    @abstractmethod
    def load(self, weight_dict):
        pass

    @abstractmethod
    def apply(self):
        pass

    def set_config(self, config={}):
        self.config = config

    def to_cuda(self, non_blocking=False):
        self.weight = self.pin_weight.cuda(non_blocking=non_blocking)
        if hasattr(self, "pin_weight_scale"):
            self.weight_scale = self.pin_weight_scale.cuda(non_blocking=non_blocking)
        if hasattr(self, "pin_bias") and self.pin_bias is not None:
            self.bias = self.pin_bias.cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pin_weight"):
            self.weight = self.pin_weight.copy_(self.weight, non_blocking=non_blocking).cpu()
            if hasattr(self, "weight_scale_name"):
                self.weight_scale = self.pin_weight_scale.copy_(self.weight_scale, non_blocking=non_blocking).cpu()
            if self.bias is not None:
                self.bias = self.pin_bias.copy_(self.bias, non_blocking=non_blocking).cpu()
        else:
            self.weight = self.weight.to("cpu", non_blocking=non_blocking)
            if hasattr(self, "weight_scale"):
                self.weight_scale = self.weight_scale.to("cpu", non_blocking=non_blocking)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias = self.bias.to("cpu", non_blocking=non_blocking)


@MM_WEIGHT_REGISTER("Default")
class MMWeight(MMWeightTemplate):
    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)

    def load(self, weight_dict):
        if self.create_cuda_buffer:
            self.weight_cuda_buffer = weight_dict[self.weight_name].t().cuda()
            if self.bias_name is not None:
                self.bias_cuda_buffer = weight_dict[self.bias_name].cuda()
        else:
            device = weight_dict[self.weight_name].device
            if device.type in ["cuda", "mlu", "npu"]:
                self.weight = weight_dict[self.weight_name].t()
                if self.bias_name is not None:
                    self.bias = weight_dict[self.bias_name]
                else:
                    self.bias = None

            elif device.type == "cpu":
                weight_shape = weight_dict[self.weight_name].shape
                weight_dtype = weight_dict[self.weight_name].dtype

                self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
                self.pin_weight = self.pin_weight.copy_(weight_dict[self.weight_name]).t()

                if self.bias_name is not None:
                    bias_shape = weight_dict[self.bias_name].shape
                    bias_dtype = weight_dict[self.bias_name].dtype
                    self.pin_bias = torch.empty(bias_shape, pin_memory=True, dtype=bias_dtype)
                    self.pin_bias.copy_(weight_dict[self.bias_name])
                else:
                    self.bias = None
                    self.pin_bias = None
                del weight_dict[self.weight_name]

            else:
                raise ValueError(f"Unsupported device type: {device.type}, only 'cpu' and 'cuda' are supported")

    def _calculate_size(self):
        if self.bias is not None:
            return self.weight.numel() * self.weight.element_size() + self.bias.numel() * self.bias.element_size()
        return self.weight.numel() * self.weight.element_size()

    def apply(self, input_tensor):
        shape = (input_tensor.shape[0], self.weight.shape[1])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)
        if self.bias is None:
            return torch.mm(input_tensor, self.weight, out=output_tensor)
        return torch.addmm(self.bias, input_tensor, self.weight, out=output_tensor)

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.weight_name] = self.pin_weight if hasattr(self, "pin_weight") else self.weight
        if self.bias_name is not None:
            destination[self.bias_name] = self.pin_bias if hasattr(self, "pin_bias") else self.bias
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        if self.is_post_adapter:
            assert adapter_block_index is not None
            weight_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.weight_name, count=1)
        else:
            weight_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_name, count=1)

        if weight_name not in destination:
            self.weight = None
            return

        self.weight = self.weight_cuda_buffer.copy_(destination[weight_name], non_blocking=True)

        if self.bias_name is not None:
            if self.is_post_adapter:
                assert adapter_block_index is not None
                bias_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.bias_name, count=1)
            else:
                bias_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.bias_name, count=1)
            self.bias = self.bias_cuda_buffer.copy_(destination[bias_name], non_blocking=True)
        else:
            self.bias = None


@MM_WEIGHT_REGISTER("Default-Force-FP32")
class MMWeightForceFP32(MMWeight):
    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)

    def load(self, weight_dict):
        super().load(weight_dict)
        self.weight = self.weight.to(torch.float32)
        if hasattr(self, "bias") and self.bias is not None:
            self.bias = self.bias.to(torch.float32)


class MMWeightQuantTemplate(MMWeightTemplate):
    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.weight_scale_name = self.weight_name.removesuffix(".weight") + ".weight_scale"
        self.load_func = None
        self.weight_need_transpose = True
        self.act_quant_func = None
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.infer_dtype = GET_DTYPE()

    # =========================
    # weight load functions
    # =========================

    def load_from_disk(self):  # Need Rewrite
        if not torch._dynamo.is_compiling():
            self.weight = self.lazy_load_file.get_tensor(self.weight_name).pin_memory()
            self.weight_scale = self.lazy_load_file.get_tensor(self.weight_scale_name).float().pin_memory()
            if self.bias_name is not None:
                self.bias = self.lazy_load_file.get_tensor(self.bias_name).to(self.infer_dtype).pin_memory()
        else:
            self.weight = self.lazy_load_file.get_tensor(self.weight_name)
            self.weight_scale = self.lazy_load_file.get_tensor(self.weight_scale_name).float()
            if self.bias_name is not None:
                self.bias = self.lazy_load_file.get_tensor(self.bias_name).to(self.infer_dtype)

        if self.weight_need_transpose:
            self.weight = self.weight.t()

    def load(self, weight_dict):
        if not self.lazy_load:
            self.load_func(weight_dict)
            if self.weight_need_transpose:
                if hasattr(self, "weight"):
                    self.weight = self.weight.t()
                if hasattr(self, "pin_weight"):
                    self.pin_weight = self.pin_weight.t()
                if hasattr(self, "weight_cuda_buffer"):
                    self.weight_cuda_buffer = self.weight_cuda_buffer.t()

    def clear(self):
        attrs = ["weight", "weight_scale", "bias", "pin_weight", "pin_weight_scale", "pin_bias"]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
                setattr(self, attr, None)

    def _calculate_size(self):
        if self.bias is not None:
            return self.weight.numel() * self.weight.element_size() + self.weight_scale.numel() * self.weight_scale.element_size() + self.bias.numel() * self.bias.element_size()
        return self.weight.numel() * self.weight.element_size() + self.weight_scale.numel() * self.weight_scale.element_size()

    def load_quantized(self, weight_dict):
        if self.create_cuda_buffer:
            # move to cuda buffer
            self.weight_cuda_buffer = weight_dict[self.weight_name].cuda()
            self.weight_scale_cuda_buffer = weight_dict[self.weight_scale_name].float().cuda()
        else:
            device = weight_dict[self.weight_name].device
            if device.type in ["cuda", "mlu", "npu"]:
                self.weight = weight_dict[self.weight_name]
                self.weight_scale = weight_dict[self.weight_scale_name].float()
            elif device.type == "cpu":
                weight_shape = weight_dict[self.weight_name].shape
                weight_dtype = weight_dict[self.weight_name].dtype
                self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
                self.pin_weight.copy_(weight_dict[self.weight_name])

                weight_scale_shape = weight_dict[self.weight_scale_name].shape
                weight_scale_dtype = torch.float
                self.pin_weight_scale = torch.empty(weight_scale_shape, pin_memory=True, dtype=weight_scale_dtype)
                self.pin_weight_scale.copy_(weight_dict[self.weight_scale_name])
                del weight_dict[self.weight_name]
            else:
                raise ValueError(f"Unsupported device type: {device.type}, only 'cpu' and 'cuda' are supported")

        if self.bias_name is not None:
            if self.create_cuda_buffer:
                # move to cuda buffer
                self.bias_cuda_buffer = weight_dict[self.bias_name].cuda()
            else:
                device = weight_dict[self.bias_name].device
                if device.type in ["cuda", "mlu", "npu"]:
                    self.bias = weight_dict[self.bias_name]
                elif device.type == "cpu":
                    bias_shape = weight_dict[self.bias_name].shape
                    bias_dtype = weight_dict[self.bias_name].dtype
                    self.pin_bias = torch.empty(bias_shape, pin_memory=True, dtype=bias_dtype)
                    self.pin_bias.copy_(weight_dict[self.bias_name])
                else:
                    raise ValueError(f"Unsupported device type: {device.type}, only 'cpu' and 'cuda' are supported")
        else:
            self.bias = None
            self.pin_bias = None

    def load_fp8_perchannel_sym(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            self.weight = weight_dict[self.weight_name].to(torch.float32)
            w_quantizer = FloatQuantizer("e4m3", True, "per_channel")
            self.weight, self.weight_scale, _ = w_quantizer.real_quant_tensor(self.weight)
            self.weight = self.weight.to(torch.float8_e4m3fn)
            self.weight_scale = self.weight_scale.to(torch.float32)
        else:
            self.load_quantized(weight_dict)

    def load_int8_perchannel_sym(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            self.weight = weight_dict[self.weight_name].to(torch.float32)
            w_quantizer = IntegerQuantizer(8, True, "per_channel")
            self.weight, self.weight_scale, _ = w_quantizer.real_quant_tensor(self.weight)
            self.weight = self.weight.to(torch.int8)
            self.weight_scale = self.weight_scale.to(torch.float32)
        else:
            self.load_quantized(weight_dict)

    def load_mxfp4(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            device = weight_dict[self.weight_name].device
            self.weight = weight_dict[self.weight_name].cuda().to(torch.bfloat16)
            self.weight, self.weight_scale = scaled_mxfp4_quant(self.weight)
            self.weight, self.weight_scale = self.weight.to(device), self.weight_scale.to(device)
        else:
            device = weight_dict[self.weight_name].device
            if device.type in ["cuda", "mlu", "npu"]:
                self.weight = weight_dict[self.weight_name]
                self.weight_scale = weight_dict[self.weight_scale_name]
            elif device.type == "cpu":
                weight_shape = weight_dict[self.weight_name].shape
                weight_dtype = weight_dict[self.weight_name].dtype
                self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
                self.pin_weight.copy_(weight_dict[self.weight_name])

                weight_scale_shape = weight_dict[self.weight_scale_name].shape
                weight_scale_dtype = weight_dict[self.weight_scale_name].dtype
                self.pin_weight_scale = torch.empty(weight_scale_shape, pin_memory=True, dtype=weight_scale_dtype)
                self.pin_weight_scale.copy_(weight_dict[self.weight_scale_name])
                del weight_dict[self.weight_name]
            else:
                raise ValueError(f"Unsupported device type: {device.type}, only 'cpu' and 'cuda' are supported")

    def load_mxfp6(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            device = weight_dict[self.weight_name].device
            self.weight = weight_dict[self.weight_name].cuda().to(torch.bfloat16)
            self.weight, self.weight_scale = scaled_mxfp6_quant(self.weight)
            self.weight, self.weight_scale = self.weight.to(device), self.weight_scale.to(device)
        else:
            device = weight_dict[self.weight_name].device
            if device.type in ["cuda", "mlu", "npu"]:
                self.weight = weight_dict[self.weight_name]
                self.weight_scale = weight_dict[self.weight_scale_name]
            elif device.type == "cpu":
                weight_shape = weight_dict[self.weight_name].shape
                weight_dtype = weight_dict[self.weight_name].dtype
                self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
                self.pin_weight.copy_(weight_dict[self.weight_name])

                weight_scale_shape = weight_dict[self.weight_scale_name].shape
                weight_scale_dtype = weight_dict[self.weight_scale_name].dtype
                self.pin_weight_scale = torch.empty(weight_scale_shape, pin_memory=True, dtype=weight_scale_dtype)
                self.pin_weight_scale.copy_(weight_dict[self.weight_scale_name])
                del weight_dict[self.weight_name]
            else:
                raise ValueError(f"Unsupported device type: {device.type}, only 'cpu' and 'cuda' are supported")

    def load_mxfp8(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            device = weight_dict[self.weight_name].device
            self.weight = weight_dict[self.weight_name].cuda().to(torch.bfloat16)
            self.weight, self.weight_scale = scaled_mxfp8_quant(self.weight)
            self.weight, self.weight_scale = self.weight.to(device), self.weight_scale.to(device)
        else:
            device = weight_dict[self.weight_name].device
            if device.type in ["cuda", "mlu", "npu"]:
                self.weight = weight_dict[self.weight_name]
                self.weight_scale = weight_dict[self.weight_scale_name]
            elif device.type == "cpu":
                weight_shape = weight_dict[self.weight_name].shape
                weight_dtype = weight_dict[self.weight_name].dtype
                self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
                self.pin_weight.copy_(weight_dict[self.weight_name])

                weight_scale_shape = weight_dict[self.weight_scale_name].shape
                weight_scale_dtype = weight_dict[self.weight_scale_name].dtype
                self.pin_weight_scale = torch.empty(weight_scale_shape, pin_memory=True, dtype=weight_scale_dtype)
                self.pin_weight_scale.copy_(weight_dict[self.weight_scale_name])
                del weight_dict[self.weight_name]
            else:
                raise ValueError(f"Unsupported device type: {device.type}, only 'cpu' and 'cuda' are supported")

    def load_nvfp4(self, weight_dict):
        device = weight_dict[self.weight_name].device

        input_absmax = weight_dict[self.weight_name.replace(".weight", ".input_absmax")]
        input_global_scale = (2688.0 / input_absmax).to(torch.float32)
        weight_global_scale = weight_dict[f"{self.weight_name}_global_scale"]
        alpha = 1.0 / (input_global_scale * weight_global_scale)

        if device.type in ["cuda", "mlu", "npu"]:
            self.weight = weight_dict[self.weight_name]
            self.weight_scale = weight_dict[self.weight_scale_name]
            self.input_global_scale = input_global_scale
            self.alpha = alpha
        elif device.type == "cpu":
            weight_shape = weight_dict[self.weight_name].shape
            weight_dtype = weight_dict[self.weight_name].dtype
            self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
            self.pin_weight.copy_(weight_dict[self.weight_name])

            weight_scale_shape = weight_dict[self.weight_scale_name].shape
            weight_scale_dtype = weight_dict[self.weight_scale_name].dtype
            self.pin_weight_scale = torch.empty(weight_scale_shape, pin_memory=True, dtype=weight_scale_dtype)
            self.pin_weight_scale.copy_(weight_dict[self.weight_scale_name])

            input_global_scale_shape = input_global_scale.shape
            input_global_scale_dtype = input_global_scale.dtype
            self.pin_input_global_scale = torch.empty(input_global_scale_shape, pin_memory=True, dtype=input_global_scale_dtype)
            self.pin_input_global_scale.copy_(input_global_scale)

            alpha_shape = alpha.shape
            alpha_dtype = alpha.dtype
            self.pin_alpha = torch.empty(alpha_shape, pin_memory=True, dtype=alpha_dtype)
            self.pin_alpha.copy_(alpha)

            del weight_dict[self.weight_name]
        else:
            raise ValueError(f"Unsupported device type: {device.type}, only 'cpu' and 'cuda' are supported")

    def load_fp8_perblock128_sym(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            self.weight = weight_dict[self.weight_name]
            self.weight, self.weight_scale = self.per_block_cast_to_fp8(self.weight)
        else:
            self.load_quantized(weight_dict)

    def per_block_cast_to_fp8(self, x):
        assert x.dim() == 2
        m, n = x.shape
        x_padded = torch.zeros(
            (deep_gemm.ceil_div(m, 128) * 128, deep_gemm.ceil_div(n, 128) * 128),
            dtype=x.dtype,
            device=x.device,
        )
        x_padded[:m, :n] = x
        x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

    # =========================
    # act quant kernels
    # =========================
    def act_quant_int8_perchannel_sym_torchao(self, x):
        input_tensor_quant, input_tensor_scale = quantize_activation_per_token_absmax(x)
        return input_tensor_quant, input_tensor_scale

    def act_quant_fp8_perchannel_sym_vllm(self, x):
        input_tensor_quant, input_tensor_scale = ops.scaled_fp8_quant(x, None, scale_ub=None, use_per_token_if_dynamic=True)
        return input_tensor_quant, input_tensor_scale

    def act_quant_fp8_perchannel_sym_sgl(self, x):
        m, k = x.shape
        input_tensor_quant = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda", requires_grad=False)
        input_tensor_scale = torch.empty((m, 1), dtype=torch.float32, device="cuda", requires_grad=False)
        sgl_kernel.sgl_per_token_quant_fp8(x, input_tensor_quant, input_tensor_scale)
        return input_tensor_quant, input_tensor_scale

    def act_quant_int8_perchannel_sym_vllm(self, x):
        input_tensor_quant, input_tensor_scale, _ = ops.scaled_int8_quant(x, scale=None, azp=None, symmetric=True)
        return input_tensor_quant, input_tensor_scale

    def act_quant_nvfp4(self, x):
        input_tensor_quant, input_tensor_scale = scaled_nvfp4_quant(x, self.input_global_scale)
        return input_tensor_quant, input_tensor_scale

    def act_quant_mxfp4(self, x):
        input_tensor_quant, input_tensor_scale = scaled_mxfp4_quant(x)
        return input_tensor_quant, input_tensor_scale

    def act_quant_mxfp8(self, x):
        input_tensor_quant, input_tensor_scale = scaled_mxfp8_quant(x)
        return input_tensor_quant, input_tensor_scale

    def act_quant_fp8_perchannelgroup128_sym_deepgemm(self, x):
        assert x.dim() == 2 and x.size(1) % 128 == 0
        m, n = x.shape
        x_view = x.view(m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

    def act_quant_fp8_perchannelgroup128_sym_sgl(self, x):
        m, k = x.shape
        input_tensor_quant = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda", requires_grad=False)
        input_tensor_scale = torch.empty((m, k // 128), dtype=torch.float32, device="cuda", requires_grad=False)
        sgl_kernel.sgl_per_token_group_quant_fp8(
            x,
            input_tensor_quant,
            input_tensor_scale,
            group_size=128,
            eps=1e-10,
            fp8_min=-448.0,
            fp8_max=448.0,
        )
        return input_tensor_quant, input_tensor_scale

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.weight_name] = self.pin_weight if hasattr(self, "pin_weight") else self.weight
        if self.bias_name is not None:
            destination[self.bias_name] = self.pin_bias if hasattr(self, "pin_bias") else self.bias
        destination[self.weight_scale_name] = self.pin_weight_scale if hasattr(self, "pin_weight_scale") else self.weight_scale
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        if self.is_post_adapter:
            weight_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.weight_name, count=1)
            weight_scale_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.weight_scale_name, count=1)
        else:
            weight_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_name, count=1)
            weight_scale_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_scale_name, count=1)

        if weight_name not in destination:
            self.weight = None
            return

        self.weight = self.weight_cuda_buffer.copy_(destination[weight_name], non_blocking=True)
        self.weight_scale = self.weight_scale_cuda_buffer.copy_(destination[weight_scale_name], non_blocking=True)

        if self.bias_name is not None:
            bias_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.bias_name, count=1)
            self.bias = self.bias_cuda_buffer.copy_(destination[bias_name], non_blocking=True)
        else:
            self.bias = None


@MM_WEIGHT_REGISTER("fp8-vllm")
class MMWeightWfp8channelAfp8channeldynamicVllm(MMWeightQuantTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: vllm
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_fp8_perchannel_sym
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_fp8_perchannel_sym_vllm

    def apply(self, input_tensor):
        shape = (input_tensor.shape[0], self.weight.shape[1])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)

        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        torch.ops._C.cutlass_scaled_mm(
            output_tensor,
            input_tensor_quant,
            self.weight,
            input_tensor_scale,
            self.weight_scale,
            self.bias if self.bias is not None else None,
        )
        return output_tensor


@MM_WEIGHT_REGISTER("int8-vllm")
class MMWeightWint8channelAint8channeldynamicVllm(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: vllm
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_int8_perchannel_sym
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_int8_perchannel_sym_vllm

    def apply(self, input_tensor):
        shape = (input_tensor.shape[0], self.weight.shape[1])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)

        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        torch.ops._C.cutlass_scaled_mm(
            output_tensor,
            input_tensor_quant,
            self.weight,
            input_tensor_scale,
            self.weight_scale,
            self.bias if self.bias is not None else None,
        )
        return output_tensor


@MM_WEIGHT_REGISTER("mxfp4")
class MMWeightWmxfp4Amxfp4dynamic(MMWeightQuantTemplate):
    """
    Name: W-mxfp4-A-mxfp4-dynamic

    Quant MM:
        Weight: mxfp4
        Act: mxfp4
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_mxfp4
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_mxfp4
        self.set_alpha()

    def set_alpha(self):
        self.alpha = torch.tensor(1.0, dtype=torch.float32)

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        self.alpha = self.alpha.to(self.weight.device)
        output_tensor = cutlass_scaled_mxfp4_mm(input_tensor_quant, self.weight, input_tensor_scale, self.weight_scale, alpha=self.alpha, bias=self.bias)
        return output_tensor


@MM_WEIGHT_REGISTER("mxfp6-mxfp8")
class MMWeightWmxfp6Amxfp8dynamic(MMWeightQuantTemplate):
    """
    Name: W-mxfp6-A-nvfp8-dynamic

    Quant MM:
        Weight: mxfp6
        Act: mxfp8
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_mxfp6
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_mxfp8
        self.set_alpha()

    def set_alpha(self):
        self.alpha = torch.tensor(1.0, dtype=torch.float32)

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        self.alpha = self.alpha.to(self.weight.device)
        output_tensor = cutlass_scaled_mxfp6_mxfp8_mm(input_tensor_quant, self.weight, input_tensor_scale, self.weight_scale, alpha=self.alpha, bias=self.bias)
        return output_tensor


@MM_WEIGHT_REGISTER("mxfp8")
class MMWeightWmxfp8Amxfp8dynamic(MMWeightQuantTemplate):
    """
    Name: W-mxfp8-A-nvfp8-dynamic

    Quant MM:
        Weight: mxfp8
        Act: mxfp8
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_mxfp8
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_mxfp8
        self.set_alpha()

    def set_alpha(self):
        self.alpha = torch.tensor(1.0, dtype=torch.float32)

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        self.alpha = self.alpha.to(self.weight.device)
        output_tensor = cutlass_scaled_mxfp8_mm(input_tensor_quant, self.weight, input_tensor_scale, self.weight_scale, alpha=self.alpha, bias=self.bias)
        return output_tensor


@MM_WEIGHT_REGISTER("nvfp4")
class MMWeightWnvfp4Anvfp4dynamic(MMWeightQuantTemplate):
    """
    Name: W-nvfp4-A-nvfp4-dynamic

    Quant MM:
        Weight: nvfp4
        Act: nvfp4
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_nvfp4
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_nvfp4

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = cutlass_scaled_nvfp4_mm(input_tensor_quant, self.weight, input_tensor_scale, self.weight_scale, alpha=self.alpha, bias=self.bias)
        return output_tensor

    def to_cuda(self, non_blocking=False):
        self.weight = self.pin_weight.cuda(non_blocking=non_blocking)
        if hasattr(self, "pin_weight_scale"):
            self.weight_scale = self.pin_weight_scale.cuda(non_blocking=non_blocking)
            self.input_global_scale = self.pin_input_global_scale.cuda(non_blocking=non_blocking)
            self.alpha = self.pin_alpha.cuda(non_blocking=non_blocking)
        if hasattr(self, "pin_bias") and self.pin_bias is not None:
            self.bias = self.pin_bias.cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pin_weight"):
            self.weight = self.pin_weight.copy_(self.weight, non_blocking=non_blocking).cpu()
            if hasattr(self, "weight_scale_name"):
                self.weight_scale = self.pin_weight_scale.copy_(self.weight_scale, non_blocking=non_blocking).cpu()
                self.input_global_scale = self.pin_input_global_scale.copy_(self.input_global_scale, non_blocking=non_blocking).cpu()
                self.alpha = self.pin_alpha.copy_(self.alpha, non_blocking=non_blocking).cpu()
            if self.bias is not None:
                self.bias = self.pin_bias.copy_(self.bias, non_blocking=non_blocking).cpu()
        else:
            self.weight = self.weight.to("cpu", non_blocking=non_blocking)
            if hasattr(self, "weight_scale"):
                self.weight_scale = self.weight_scale.to("cpu", non_blocking=non_blocking)
                self.input_global_scale = self.input_global_scale.to("cpu", non_blocking=non_blocking)
                self.alpha = self.alpha.to("cpu", non_blocking=non_blocking)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias = self.bias.to("cpu", non_blocking=non_blocking)


@MM_WEIGHT_REGISTER("Calib")
class MMCalibNvfp4(MMWeight):
    """
    Name: calib

    Calib:
        absmax: torch.max(torch.abs(input_tensor))
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.running_absmax = None
        self.count = 0
        self.decay = 0.9

    def apply(self, input_tensor):
        shape = (input_tensor.shape[0], self.weight.shape[1])
        dtype, device = input_tensor.dtype, input_tensor.device

        current_absmax = torch.max(torch.abs(input_tensor)).to("cpu")
        if self.count % 2 == 0:
            if self.running_absmax is None:
                self.running_absmax = current_absmax
            else:
                self.running_absmax = self.decay * self.running_absmax + (1 - self.decay) * current_absmax
            CALIB["absmax"][self.weight_name] = self.running_absmax
        self.count = self.count + 1

        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)
        if self.bias is None:
            return torch.mm(input_tensor, self.weight, out=output_tensor)
        return torch.addmm(self.bias, input_tensor, self.weight, out=output_tensor)


@MM_WEIGHT_REGISTER("fp8-q8f")
class MMWeightWfp8channelAfp8channeldynamicQ8F(MMWeightQuantTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Q8F

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: Q8F
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_fp8_perchannel_sym
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_fp8_perchannel_sym_vllm

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = fp8_linear(
            input_tensor_quant,
            self.weight,
            self.bias.float() if self.bias is not None else None,
            input_tensor_scale,
            self.weight_scale,
            out_dtype=self.infer_dtype,
        )
        return output_tensor.squeeze(0) if len(output_tensor.shape) == 3 else output_tensor


@MM_WEIGHT_REGISTER("int8-q8f")
class MMWeightWint8channelAint8channeldynamicQ8F(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Q8F

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: Q8F
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_int8_perchannel_sym
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_int8_perchannel_sym_vllm

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = q8_linear(
            input_tensor_quant,
            self.weight,
            self.bias.float() if self.bias is not None else None,
            input_tensor_scale,
            self.weight_scale,
            fuse_gelu=False,
            out_dtype=self.infer_dtype,
        )
        return output_tensor.squeeze(0) if len(output_tensor.shape) == 3 else output_tensor


@MM_WEIGHT_REGISTER("fp8-b128-deepgemm")
class MMWeightWfp8block128Afp8channelgroup128dynamicDeepgemmActSgl(MMWeightQuantTemplate):
    """
    Name: W-fp8-block128-sym-A-fp8-channel-group128-sym-dynamic-Deepgemm-ActSgl

    Quant MM:
        Weight: fp8 perblock 128x128 sym
        Act: fp8 pertoken-pergroup group=128 dynamic sym
        Kernel: quant-mm using Deepgemm, act dynamic quant using Sgl-kernel
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_fp8_perblock128_sym
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_fp8_perchannelgroup128_sym_sgl

    def apply(self, input_tensor):
        shape = (input_tensor.shape[0], self.weight.shape[0])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)

        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        deep_gemm.gemm_fp8_fp8_bf16_nt(
            (input_tensor_quant, input_tensor_scale),
            (self.weight, self.weight_scale),
            output_tensor,
        )
        if hasattr(self, "bias") and self.bias is not None:
            output_tensor.add_(self.bias)
        return output_tensor


@MM_WEIGHT_REGISTER("fp8-sgl")
class MMWeightWfp8channelAfp8channeldynamicSgl(MMWeightQuantTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Sgl

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: Sgl-kernel
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_fp8_perchannel_sym
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_fp8_perchannel_sym_sgl

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = sgl_kernel.fp8_scaled_mm(
            input_tensor_quant,
            self.weight,
            input_tensor_scale,
            self.weight_scale,
            self.infer_dtype,
            bias=self.bias,
        )
        return output_tensor


@MM_WEIGHT_REGISTER("int8-sgl")
class MMWeightWint8channelAint8channeldynamicSglActVllm(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Sgl-ActVllm

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: quant-mm using Sgl-kernel, act dynamic quant using vllm
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_int8_perchannel_sym
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_int8_perchannel_sym_vllm

    def apply(self, input_tensor):
        shape = (input_tensor.shape[0], self.weight.shape[1])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)

        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = sgl_kernel.int8_scaled_mm(
            input_tensor_quant,
            self.weight,
            input_tensor_scale,
            self.weight_scale,
            self.infer_dtype,
            self.bias if self.bias is not None else None,
        )
        return output_tensor


@MM_WEIGHT_REGISTER("int8-torchao")
class MMWeightWint8channelAint8channeldynamicSglActVllm(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Torchao

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: Torchao
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_int8_perchannel_sym
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_int8_perchannel_sym_torchao

    def apply(self, input_tensor):
        input_tensor = input_tensor
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = quant_int8_per_token_matmul(input_tensor_quant, input_tensor_scale, self.weight, self.weight_scale.t().float(), output_dtype=self.infer_dtype)
        if self.bias is not None:
            output_tensor = output_tensor + self.bias

        return output_tensor


class MMWeightGGUFTemplate(MMWeightTemplate):
    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)

    def load(self, weight_dict):
        assert not self.create_cuda_buffer, "GGUF Unsupported offload block"
        self.weight = weight_dict[self.weight_name]

        weight_shape = self.weight.shape
        weight_dtype = self.weight.dtype

        if isinstance(self.weight, GGMLTensor):
            self.pin_weight = GGMLTensor.empty_pinned(weight_shape, orig_shape=self.weight.orig_shape, dtype=weight_dtype, gguf_type=self.weight.gguf_type)
            self.pin_weight.copy_from(self.weight)
        else:
            self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
            self.pin_weight.copy_(weight_dict[self.weight_name])

        if self.bias_name is not None:
            self.bias = weight_dict[self.bias_name]
            if isinstance(self.bias, GGMLTensor):
                self.pin_bias = GGMLTensor.empty_pinned(self.bias.shape, orig_shape=self.bias.orig_shape, dtype=self.bias.dtype, gguf_type=self.bias.gguf_type)
                self.pin_bias.copy_from(self.bias)
            else:
                self.pin_bias = torch.empty(self.bias.shape, pin_memory=True, dtype=self.bias.dtype)
                self.pin_bias.copy_(weight_dict[self.bias_name])
        else:
            self.bias = None

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        if self.is_post_adapter:
            assert adapter_block_index is not None
            weight_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.weight_name, count=1)
        else:
            weight_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_name, count=1)

        if weight_name not in destination:
            self.weight = None
            return

        self.weight = self.weight_cuda_buffer.copy_(destination[weight_name], non_blocking=True)

        if self.bias_name is not None:
            if self.is_post_adapter:
                assert adapter_block_index is not None
                bias_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.bias_name, count=1)
            else:
                bias_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.bias_name, count=1)
            self.bias = self.bias_cuda_buffer.copy_(destination[bias_name], non_blocking=True)
        else:
            self.bias = None

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.weight_name] = self.pin_weight if hasattr(self, "pin_weight") else self.weight
        if self.bias_name is not None:
            destination[self.bias_name] = self.pin_bias if hasattr(self, "pin_bias") else self.bias

        return destination

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return

        device = tensor.device
        weight = gguf_dequantize_tensor(tensor, dtype)
        # prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)

        return weight

    def cast_bias_weight(self, input_tensor=None, dtype=None, device=None, bias_dtype=None):
        if input_tensor is not None:
            if dtype is None:
                dtype = getattr(input_tensor, "dtype", torch.float32)

        bias = None
        if self.bias is not None:
            bias = self.get_weight(self.bias, dtype)

        weight = self.get_weight(self.weight, dtype)
        return weight, bias

    def apply(self, input_tensor):
        weight, bias = self.cast_bias_weight(input_tensor)
        return torch.nn.functional.linear(input_tensor, weight, bias)


@MM_WEIGHT_REGISTER("gguf-BF16")
class MMWeightGGUFBF16(MMWeightGGUFTemplate):
    qtype = gguf.GGMLQuantizationType.BF16


@MM_WEIGHT_REGISTER("gguf-Q8_0")
class MMWeightGGUFQ80(MMWeightGGUFTemplate):
    qtype = gguf.GGMLQuantizationType.Q8_0


@MM_WEIGHT_REGISTER("gguf-Q6_K")
class MMWeightGGUFQ6K(MMWeightGGUFTemplate):
    qtype = gguf.GGMLQuantizationType.Q6_K


@MM_WEIGHT_REGISTER("gguf-Q5_K_S")
class MMWeightGGUFQ5KS(MMWeightGGUFTemplate):
    qtype = gguf.GGMLQuantizationType.Q6_K


@MM_WEIGHT_REGISTER("gguf-Q5_K_M")
class MMWeightGGUFQ5KM(MMWeightGGUFTemplate):
    qtype = gguf.GGMLQuantizationType.Q6_K


@MM_WEIGHT_REGISTER("gguf-Q5_1")
class MMWeightGGUFQ51(MMWeightGGUFTemplate):
    qtype = gguf.GGMLQuantizationType.Q5_1


@MM_WEIGHT_REGISTER("gguf-Q5_0")
class MMWeightGGUFQ50(MMWeightGGUFTemplate):
    qtype = gguf.GGMLQuantizationType.Q5_0


@MM_WEIGHT_REGISTER("gguf-Q4_K_M")
class MMWeightGGUFQ4KM(MMWeightGGUFTemplate):
    qtype = gguf.GGMLQuantizationType.Q5_0


@MM_WEIGHT_REGISTER("gguf-Q4_K_S")
class MMWeightGGUFQ4KS(MMWeightGGUFTemplate):
    qtype = gguf.GGMLQuantizationType.Q4_K


@MM_WEIGHT_REGISTER("gguf-Q4_1")
class MMWeightGGUFQ41(MMWeightGGUFTemplate):
    qtype = gguf.GGMLQuantizationType.Q4_1


@MM_WEIGHT_REGISTER("gguf-Q4_0")
class MMWeightGGUFQ40(MMWeightGGUFTemplate):
    qtype = gguf.GGMLQuantizationType.Q4_0


@MM_WEIGHT_REGISTER("gguf-Q3_K_M")
class MMWeightGGUFQ3KM(MMWeightGGUFTemplate):
    qtype = gguf.GGMLQuantizationType.Q3_K


@MM_WEIGHT_REGISTER("gguf-Q3_K_S")
class MMWeightGGUFQ3KS(MMWeightGGUFTemplate):
    qtype = gguf.GGMLQuantizationType.Q2_K


@MM_WEIGHT_REGISTER("int4-g128-marlin")
class MMWeightWint4group128Marlin(MMWeightQuantTemplate):
    """
    Name: "W-int4-group128-sym-Marlin

    Quant int4 x FP16:
        Weight: int4 pergroup sym
        Kernel: Marlin
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_quantized

    def load(self, weight_dict):
        assert not self.lazy_load
        self.load_func(weight_dict)
        self.workspace = weight_dict[f"{self.weight_name}_workspace"]

        if self.bias_name is not None:
            bias_shape = weight_dict[self.bias_name].shape
            bias_dtype = weight_dict[self.bias_name].dtype
            self.bias = torch.empty(bias_shape, pin_memory=True, dtype=bias_dtype)
            self.bias.copy_(weight_dict[self.bias_name])
        else:
            self.bias = None

    def apply(self, input_tensor):
        output_tensor = torch.empty(input_tensor.shape[:-1] + (self.weight_scale.shape[1],), dtype=input_tensor.dtype, device=input_tensor.device)
        marlin_cuda_quant.mul(input_tensor, self.weight, output_tensor, self.weight_scale.half(), self.workspace, -1, -1, -1, -1)
        if hasattr(self, "bias") and self.bias is not None:
            output_tensor.add_(self.bias)
        return output_tensor


@MM_WEIGHT_REGISTER("int8-tmo")
class MMWeightWint8channelAint8channeldynamicMlu(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Mlu

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: mlu
    """

    def __init__(self, weight_name, bias_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_int8_perchannel_sym
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_int8_perchannel_sym_tmo

    def act_quant_int8_perchannel_sym_tmo(self, x):
        input_tensor_quant, input_tensor_scale = tmo.scaled_quantize(x)
        return input_tensor_quant, input_tensor_scale

    def apply(self, input_tensor):
        dtype = input_tensor.dtype
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = tmo.scaled_matmul(input_tensor_quant, self.weight.contiguous(), input_tensor_scale, self.weight_scale.squeeze(-1), output_dtype=dtype, use_hp_active=True)
        return output_tensor
