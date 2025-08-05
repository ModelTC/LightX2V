from abc import ABCMeta, abstractmethod

import torch
from loguru import logger

from lightx2v.utils.envs import *
from lightx2v.utils.quant_utils import FloatQuantizer, IntegerQuantizer
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER

try:
    from vllm import _custom_ops as ops
except ImportError:
    ops = None

try:
    import sgl_kernel
except ImportError:
    sgl_kernel = None

try:
    import q8_kernels.functional as Q8F
except ImportError:
    Q8F = None

try:
    import deep_gemm
except ImportError:
    deep_gemm = None

try:
    from torchao.quantization.utils import quant_int8_per_token_matmul, quantize_activation_per_token_absmax
except ModuleNotFoundError:
    quant_int8_per_token_matmul, quantize_activation_per_token_absmax = None, None

try:
    import gguf
except ImportError:
    gguf = None


class MMWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.config = {}

    @abstractmethod
    def load(self, weight_dict):
        pass

    @abstractmethod
    def apply(self, input_tensor):
        pass

    def set_config(self, config={}):
        self.config = config

    def to_cuda(self, non_blocking=False):
        self.weight = self.weight.cuda(non_blocking=non_blocking)
        if hasattr(self, "weight_scale"):
            self.weight_scale = self.weight_scale.cuda(non_blocking=non_blocking)
        if hasattr(self, "bias") and self.bias is not None:
            self.bias = self.bias.cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pinned_weight"):
            self.weight = self.pinned_weight.copy_(self.weight, non_blocking=non_blocking).cpu()
            if hasattr(self, "weight_scale_name"):
                self.weight_scale = self.pinned_weight_scale.copy_(self.weight_scale, non_blocking=non_blocking).cpu()
            if self.bias is not None:
                self.bias = self.pinned_bias.copy_(self.bias, non_blocking=non_blocking).cpu()
        else:
            self.weight = self.weight.to("cpu", non_blocking=non_blocking)
            if hasattr(self, "weight_scale"):
                self.weight_scale = self.weight_scale.to("cpu", non_blocking=non_blocking)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias = self.bias.to("cpu", non_blocking=non_blocking)


@MM_WEIGHT_REGISTER("Default")
class MMWeight(MMWeightTemplate):
    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)

    def load(self, weight_dict):
        self.weight = weight_dict[self.weight_name].t()
        self.pinned_weight = torch.empty(self.weight.shape, pin_memory=True, dtype=self.weight.dtype)
        self.bias = weight_dict[self.bias_name] if self.bias_name is not None else None
        self.pinned_bias = torch.empty(self.bias.shape, pin_memory=True, dtype=self.bias.dtype) if self.bias is not None else None

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
        destination[self.weight_name] = self.weight.cpu().detach().clone().t().contiguous()
        if hasattr(self, "bias") and self.bias is not None:
            destination[self.bias_name] = self.bias.cpu().detach().clone()
        return destination


@MM_WEIGHT_REGISTER("Default-Force-FP32")
class MMWeightForceFP32(MMWeight):
    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)

    def load(self, weight_dict):
        super().load(weight_dict)
        self.weight = self.weight.to(torch.float32)
        if hasattr(self, "bias") and self.bias is not None:
            self.bias = self.bias.to(torch.float32)


class MMWeightQuantTemplate(MMWeightTemplate):
    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)
        self.weight_scale_name = self.weight_name.removesuffix(".weight") + ".weight_scale"
        self.load_func = None
        self.weight_need_transpose = True
        self.act_quant_func = None
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

    # =========================
    # weight load functions
    # =========================

    def load_from_disk(self):
        if not torch._dynamo.is_compiling():
            self.weight = self.lazy_load_file.get_tensor(self.weight_name).pin_memory()
            self.weight_scale = self.lazy_load_file.get_tensor(self.weight_scale_name).float().pin_memory()
            if self.bias_name is not None:
                self.bias = self.lazy_load_file.get_tensor(self.bias_name).to(torch.bfloat16).pin_memory()
        else:
            self.weight = self.lazy_load_file.get_tensor(self.weight_name)
            self.weight_scale = self.lazy_load_file.get_tensor(self.weight_scale_name).float()
            if self.bias_name is not None:
                self.bias = self.lazy_load_file.get_tensor(self.bias_name).to(torch.bfloat16)

        if self.weight_need_transpose:
            self.weight = self.weight.t()

    def load(self, weight_dict):
        if not self.lazy_load:
            self.load_func(weight_dict)
            if self.weight_need_transpose:
                self.weight = self.weight.t()
                self.pinned_weight = self.pinned_weight.t()

    def clear(self):
        attrs = ["weight", "weight_scale", "bias", "pinned_weight", "pinned_weight_scale", "pinned_bias"]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
                setattr(self, attr, None)

    def _calculate_size(self):
        if self.bias is not None:
            return self.weight.numel() * self.weight.element_size() + self.weight_scale.numel() * self.weight_scale.element_size() + self.bias.numel() * self.bias.element_size()

        return self.weight.numel() * self.weight.element_size() + self.weight_scale.numel() * self.weight_scale.element_size()

    def load_quantized(self, weight_dict):
        self.weight = weight_dict[self.weight_name]
        self.weight_scale = weight_dict[self.weight_scale_name].float()

        self.pinned_weight = torch.empty(self.weight.shape, pin_memory=True, dtype=self.weight.dtype)
        self.pinned_weight_scale = torch.empty(self.weight_scale.shape, pin_memory=True, dtype=self.weight_scale.dtype)

    def load_fp8_perchannel_sym(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            self.weight = weight_dict[self.weight_name].to(torch.float32)
            w_quantizer = FloatQuantizer("e4m3", True, "per_channel")
            self.weight, self.weight_scale, _ = w_quantizer.real_quant_tensor(self.weight)
            self.weight = self.weight.to(torch.float8_e4m3fn)
            self.weight_scale = self.weight_scale.to(torch.float32)
            self.pinned_weight = torch.empty(self.weight.shape, pin_memory=True, dtype=self.weight.dtype)
            self.pinned_weight_scale = torch.empty(self.weight_scale.shape, pin_memory=True, dtype=self.weight_scale.dtype)
        else:
            self.load_quantized(weight_dict)

        if self.bias_name is not None:
            self.bias = weight_dict[self.bias_name]
            self.pinned_bias = torch.empty(self.bias.shape, pin_memory=True, dtype=self.bias.dtype)
        else:
            self.bias = None

    def load_int8_perchannel_sym(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            self.weight = weight_dict[self.weight_name].to(torch.float32)
            w_quantizer = IntegerQuantizer(8, True, "per_channel")
            self.weight, self.weight_scale, _ = w_quantizer.real_quant_tensor(self.weight)
            self.weight = self.weight.to(torch.int8)
            self.weight_scale = self.weight_scale.to(torch.float32)
            self.pinned_weight = torch.empty(self.weight.shape, pin_memory=True, dtype=self.weight.dtype)
            self.pinned_weight_scale = torch.empty(self.weight_scale.shape, pin_memory=True, dtype=self.weight_scale.dtype)
        else:
            self.load_quantized(weight_dict)

        if self.bias_name is not None:
            self.bias = weight_dict[self.bias_name]
            self.pinned_bias = torch.empty(self.bias.shape, pin_memory=True, dtype=self.bias.dtype)
        else:
            self.bias = None

    def load_fp8_perblock128_sym(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            self.weight = weight_dict[self.weight_name]
            self.weight, self.weight_scale = self.per_block_cast_to_fp8(self.weight)
            self.pinned_weight = torch.empty(self.weight.shape, pin_memory=True, dtype=self.weight.dtype)
            self.pinned_weight_scale = torch.empty(self.weight_scale.shape, pin_memory=True, dtype=self.weight_scale.dtype)
        else:
            self.load_quantized(weight_dict)

        if self.bias_name is not None:
            self.bias = weight_dict[self.bias_name]
            self.pinned_bias = torch.empty(self.bias.shape, pin_memory=True, dtype=self.bias.dtype)
        else:
            self.bias = None

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
        if self.weight_need_transpose:
            destination[self.weight_name] = self.weight.cpu().detach().clone().t().contiguous()
        else:
            destination[self.weight_name] = self.weight.cpu().detach().clone().contiguous()
        if hasattr(self, "bias") and self.bias is not None:
            destination[self.bias_name] = self.bias.cpu().detach().clone()
        if hasattr(self, "weight_scale"):
            destination[self.weight_name.removesuffix(".weight") + ".weight_scale"] = self.weight_scale.cpu().detach().clone()
        return destination


@MM_WEIGHT_REGISTER("W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm")
class MMWeightWfp8channelAfp8channeldynamicVllm(MMWeightQuantTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: vllm
    """

    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)
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
            self.bias,
        )
        return output_tensor


@MM_WEIGHT_REGISTER("W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm")
class MMWeightWint8channelAint8channeldynamicVllm(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: vllm
    """

    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)
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
            self.bias,
        )
        return output_tensor


@MM_WEIGHT_REGISTER("W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Q8F")
class MMWeightWfp8channelAfp8channeldynamicQ8F(MMWeightQuantTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Q8F

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: Q8F
    """

    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)
        self.load_func = self.load_fp8_perchannel_sym
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_fp8_perchannel_sym_vllm

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = Q8F.linear.fp8_linear(
            input_tensor_quant,
            self.weight,
            self.bias.float(),
            input_tensor_scale,
            self.weight_scale,
            out_dtype=torch.bfloat16,
        )
        return output_tensor.squeeze(0)


@MM_WEIGHT_REGISTER("W-int8-channel-sym-A-int8-channel-sym-dynamic-Q8F")
class MMWeightWint8channelAint8channeldynamicQ8F(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Q8F

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: Q8F
    """

    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)
        self.load_func = self.load_int8_perchannel_sym
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_int8_perchannel_sym_vllm

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = Q8F.linear.q8_linear(
            input_tensor_quant,
            self.weight,
            self.bias.float(),
            input_tensor_scale,
            self.weight_scale,
            fuse_gelu=False,
            out_dtype=torch.bfloat16,
        )
        return output_tensor.squeeze(0)


@MM_WEIGHT_REGISTER("W-fp8-block128-sym-A-fp8-channel-group128-sym-dynamic-Deepgemm")
class MMWeightWfp8block128Afp8channelgroup128dynamicDeepgemm(MMWeightQuantTemplate):
    """
    Name: W-fp8-block128-sym-A-fp8-channel-group128-sym-dynamic-Deepgemm

    Quant MM:
        Weight: fp8 perblock 128x128 sym
        Act: fp8 perchannel-pergroup group=128 dynamic sym
        Kernel: Deepgemm

    Reference: https://github.com/deepseek-ai/DeepGEMM

    Example:
        Act(1024, 2048) x Weight(2048, 4096) = Out(1024, 4096)

        Act : torch.Size([1024, 2048]), torch.float8_e4m3fn
        Act Scale: torch.Size([1024, 16]), torch.float32
        Weight : torch.Size([4096, 2048]), torch.float8_e4m3fn
        Weight Scale: torch.Size([32, 16]), torch.float32
        Out : torch.Size([1024, 4096]), torch.bfloat16
    """

    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)
        self.load_func = self.load_fp8_perblock128_sym
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_fp8_perchannelgroup128_sym_deepgemm

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


@MM_WEIGHT_REGISTER("W-fp8-block128-sym-A-fp8-channel-group128-sym-dynamic-Deepgemm-ActSgl")
class MMWeightWfp8block128Afp8channelgroup128dynamicDeepgemmActSgl(MMWeightQuantTemplate):
    """
    Name: W-fp8-block128-sym-A-fp8-channel-group128-sym-dynamic-Deepgemm-ActSgl

    Quant MM:
        Weight: fp8 perblock 128x128 sym
        Act: fp8 pertoken-pergroup group=128 dynamic sym
        Kernel: quant-mm using Deepgemm, act dynamic quant using Sgl-kernel
    """

    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)
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


@MM_WEIGHT_REGISTER("W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm-ActSgl")
class MMWeightWfp8channelAfp8channeldynamicVllmActSgl(MMWeightQuantTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm-ActSgl

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: quant-mm using vllm, act dynamic quant using Sgl-kernel
    """

    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)
        self.load_func = self.load_fp8_perchannel_sym
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_fp8_perchannel_sym_sgl

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
            self.bias,
        )
        return output_tensor


@MM_WEIGHT_REGISTER("W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Sgl-ActVllm")
class MMWeightWfp8channelAfp8channeldynamicSglActVllm(MMWeightQuantTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Sgl-ActVllm

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: quant-mm using Sgl-kernel, act dynamic quant using vllm
    """

    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)
        self.load_func = self.load_fp8_perchannel_sym
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_fp8_perchannel_sym_vllm

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = sgl_kernel.fp8_scaled_mm(
            input_tensor_quant,
            self.weight,
            input_tensor_scale,
            self.weight_scale,
            torch.bfloat16,
            bias=self.bias,
        )
        return output_tensor


@MM_WEIGHT_REGISTER("W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Sgl")
class MMWeightWfp8channelAfp8channeldynamicSgl(MMWeightQuantTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Sgl

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: Sgl-kernel
    """

    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)
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
            torch.bfloat16,
            bias=self.bias,
        )
        return output_tensor


@MM_WEIGHT_REGISTER("W-int8-channel-sym-A-int8-channel-sym-dynamic-Sgl-ActVllm")
class MMWeightWint8channelAint8channeldynamicSglActVllm(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Sgl-ActVllm

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: quant-mm using Sgl-kernel, act dynamic quant using vllm
    """

    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)
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
            torch.bfloat16,
            self.bias,
        )
        return output_tensor


@MM_WEIGHT_REGISTER("W-int8-channel-sym-A-int8-channel-sym-dynamic-Torchao")
class MMWeightWint8channelAint8channeldynamicSglActVllm(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Torchao

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: Torchao
    """

    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)
        self.load_func = self.load_int8_perchannel_sym
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_int8_perchannel_sym_torchao

    def apply(self, input_tensor):
        input_tensor = input_tensor
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = quant_int8_per_token_matmul(input_tensor_quant, input_tensor_scale, self.weight, self.weight_scale.t().float(), output_dtype=torch.bfloat16)
        if self.bias is not None:
            output_tensor = output_tensor + self.bias

        return output_tensor


class MMWeightGGUFTemplate(MMWeightQuantTemplate):
    TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)

    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)

    def dequantize_func(self):
        # TODO: implement dequantize_func
        pass


@MM_WEIGHT_REGISTER("W-gguf-Q4_K")
class MMWeightGGUFQ4K(MMWeightGGUFTemplate):
    def __init__(self, weight_name, bias_name, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file)


if __name__ == "__main__":
    weight_dict = {
        "xx.weight": torch.randn(8192, 4096).to(torch.float8_e4m3fn),
        "xx.bias": torch.randn(8192).to(torch.bfloat16),
        "xx.weight_scale": torch.randn(8192, 1).to(torch.float32),
    }

    mm_weight = MM_WEIGHT_REGISTER["W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"]("xx.weight", "xx.bias")
    mm_weight.set_config({"weight_auto_quant": False})
    mm_weight.load(weight_dict)
    input_tensor = torch.randn(1024, 4096).to(torch.bfloat16).cuda()
    output_tensor = mm_weight.apply(input_tensor)
    logger.info(output_tensor.shape)

    weight_dict = {
        "xx.weight": torch.randn(8192, 4096),
        "xx.bias": torch.randn(8192).to(torch.bfloat16),
    }

    mm_weight = MM_WEIGHT_REGISTER["W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm"]("xx.weight", "xx.bias")
    mm_weight.set_config({"weight_auto_quant": True})
    mm_weight.load(weight_dict)
    input_tensor = torch.randn(1024, 4096).to(torch.bfloat16).cuda()
    output_tensor = mm_weight.apply(input_tensor)
    logger.info(output_tensor.shape)

    weight_dict = {
        "xx.weight": torch.randn(8192, 4096),
        "xx.bias": torch.randn(8192).to(torch.bfloat16),
    }

    mm_weight = MM_WEIGHT_REGISTER["W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm"]("xx.weight", "xx.bias")
    mm_weight.set_config({"weight_auto_quant": True})
    mm_weight.load(weight_dict)
    input_tensor = torch.randn(1024, 4096).to(torch.bfloat16).cuda()
    output_tensor = mm_weight.apply(input_tensor)
    logger.info(output_tensor.shape)
