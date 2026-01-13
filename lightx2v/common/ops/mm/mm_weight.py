import re
from abc import ABCMeta, abstractmethod

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.common.ops.mm.triton_kernels import (
    fp8_gemm_bias_triton,
    fp8_gemm_triton,
    fp8_quantize_triton,
    int8_gemm_bias_triton,
    int8_gemm_triton,
    int8_quantize_triton,
)
from lightx2v.common.ops.utils import *
from lightx2v.utils.envs import *
from lightx2v.utils.ggml_tensor import GGMLTensor
from lightx2v.utils.ggml_tensor import dequantize_tensor as gguf_dequantize_tensor
from lightx2v.utils.global_paras import CALIB
from lightx2v.utils.quant_utils import FloatQuantizer, IntegerQuantizer
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

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
    from torchao.quantization.utils import (
        quant_int8_per_token_matmul as torchao_int8_gemm,
    )
    from torchao.quantization.utils import (
        quantize_activation_per_token_absmax as torchao_int8_quant,
    )
except ImportError:
    try:
        from torchao.quantization.utils import (
            _quant_int8_per_token_matmul as torchao_int8_gemm,
        )
        from torchao.quantization.utils import (
            _quantize_activation_per_token_absmax as torchao_int8_quant,
        )
    except ImportError:
        torchao_int8_gemm, torchao_int8_quant = None, None

try:
    import gguf
except ImportError:
    gguf = None

try:
    import marlin_cuda_quant
except ImportError:
    marlin_cuda_quant = None


class MMWeightTemplate(metaclass=ABCMeta):
    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.create_cuda_buffer = create_cuda_buffer
        self.create_cpu_buffer = create_cpu_buffer
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.is_post_adapter = is_post_adapter
        self.config = {}
        self.lora_prefix = lora_prefix
        self.lora_path = lora_path
        self.has_lora_branch = False
        self.has_diff = False
        self._get_base_attrs_mapping()
        self._get_lora_attr_mapping()

    def _get_base_attrs_mapping(self):
        self.base_attrs = [
            (self.weight_name, "weight", True),
        ]
        if self.bias_name is not None:
            self.base_attrs.append((self.bias_name, "bias", False))

    def _get_lora_attr_mapping(self):
        self.lora_down_name, self.lora_up_name, self.lora_alpha_name, self.weight_diff_name, self.bias_diff_name = build_lora_and_diff_names(self.weight_name, self.lora_prefix)
        self.lora_attrs = {
            "lora_alpha": "lora_alpha_name",
            "lora_down": "lora_down_name",
            "lora_up": "lora_up_name",
            "weight_diff": "weight_diff_name",
            "bias_diff": "bias_diff_name",
        }

    def _get_actual_weight(self):
        if not hasattr(self, "weight_diff"):
            return self.weight
        return self.weight + self.weight_diff

    def _get_actual_bias(self, bias=None):
        if bias is not None:
            if not hasattr(self, "bias_diff"):
                return bias
            return bias + self.bias_diff
        else:
            if self.bias is None:
                return None
            if not hasattr(self, "bias_diff"):
                return self.bias
            return self.bias + self.bias_diff

    def apply_lora(self, input_tensor):
        h = torch.mm(input_tensor, self.lora_down.t())
        out = torch.mm(h, self.lora_up.t())
        return self.lora_strength * self.lora_scale * out

    def set_config(self, config={}):
        self.config = config

    def register_diff(self, weight_dict):
        if not self.lazy_load or self.create_cuda_buffer or self.create_cpu_buffer:
            if self.weight_diff_name in weight_dict:
                self.has_diff = True
                self.weight_diff = weight_dict[self.weight_diff_name].t()
                logger.debug(f"Register Diff to {self.weight_name}")
            if self.bias_diff_name in weight_dict:
                self.has_diff = True
                self.bias_diff = weight_dict[self.bias_diff_name]
                logger.debug(f"Register Diff to {self.bias_name}")

    def register_lora(self, weight_dict, lora_strength=1):
        if not self.lazy_load or self.create_cuda_buffer or self.create_cpu_buffer:
            if self.lora_down_name in weight_dict:
                self.has_lora_branch = True
                self.lora_down = weight_dict[self.lora_down_name]
                self.lora_up = weight_dict[self.lora_up_name]
                self.lora_strength = lora_strength
                if self.lora_alpha_name in weight_dict:
                    self.lora_alpha = weight_dict[self.lora_alpha_name]
                    self.lora_scale = self.lora_alpha / self.lora_down.shape[0]
                else:
                    self.lora_scale = torch.tensor(1.0, device=AI_DEVICE)
                logger.debug(f"Register LoRA to {self.weight_name} with lora_scale={self.lora_scale}")

    def state_dict(self, destination=None):
        return state_dict(self, self.base_attrs, self.lora_attrs, destination)

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        return load_state_dict(self, self.base_attrs, self.lora_attrs, destination, block_index, adapter_block_index)

    def load_lora_state_dict_from_disk(self, block_index):
        self.lora_alpha_name = resolve_block_name(self.lora_alpha_name, block_index)
        self.lora_down_name = resolve_block_name(self.lora_down_name, block_index)
        self.lora_up_name = resolve_block_name(self.lora_up_name, block_index)
        self.weight_diff_name = resolve_block_name(self.weight_diff_name, block_index)
        self.bias_diff_name = resolve_block_name(self.bias_diff_name, block_index)
        with safe_open(self.lora_path, framework="pt", device="cpu") as lora_load_file:
            for lora_attr, lora_attr_name in self.lora_attrs.items():
                if getattr(self, lora_attr_name) in lora_load_file.keys():
                    setattr(self, lora_attr, getattr(self, lora_attr).copy_(lora_load_file.get_tensor(getattr(self, lora_attr_name)), non_blocking=True))

    def to_cuda(self, non_blocking=False):
        move_attr_to_cuda(self, self.base_attrs, self.lora_attrs, non_blocking)

    def to_cpu(self, non_blocking=False):
        move_attr_to_cpu(self, self.base_attrs, self.lora_attrs, non_blocking)

    @abstractmethod
    def load(self, weight_dict):
        pass

    @abstractmethod
    def apply(self):
        pass


@MM_WEIGHT_REGISTER("Default")
class MMWeight(MMWeightTemplate):
    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )

    def load(self, weight_dict):
        if not self.create_cuda_buffer and not self.create_cpu_buffer and not self.lazy_load:
            device_tensors, pin_tensors = create_default_tensors(self.base_attrs, weight_dict)
            self.weight = device_tensors.get("weight")
            self.bias = device_tensors.get("bias")
            self.pin_weight = pin_tensors.get("weight")
            self.pin_bias = pin_tensors.get("bias")
        elif self.create_cuda_buffer:
            result = create_cuda_buffers(self.base_attrs, weight_dict, self.lazy_load, self.lazy_load_file)
            self.weight_cuda_buffer = result.get("weight")
            self.bias_cuda_buffer = result.get("bias")
        elif self.create_cpu_buffer:
            result = create_cpu_buffers(self.base_attrs, self.lazy_load_file)
            self.pin_weight = result.get("weight")
            self.pin_bias = result.get("bias")
            self.weight = None
            self.bias = None

    def apply(self, input_tensor):
        shape = (input_tensor.shape[0], self.weight.shape[1])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)

        if not self.has_lora_branch:
            if hasattr(self, "bias") and self.bias is not None:
                return torch.addmm(self._get_actual_bias(), input_tensor, self._get_actual_weight(), out=output_tensor)
            return torch.mm(input_tensor, self._get_actual_weight(), out=output_tensor)
        else:
            if hasattr(self, "bias") and self.bias is not None:
                return torch.addmm(self._get_actual_bias(), input_tensor, self._get_actual_weight(), out=output_tensor) + self.apply_lora(input_tensor)
            return torch.mm(input_tensor, self._get_actual_weight(), out=output_tensor) + self.apply_lora(input_tensor)

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        if self.has_lora_branch or self.has_diff:
            self.load_lora_state_dict_from_disk(block_index)
        self.weight_name = resolve_block_name(self.weight_name, block_index, adapter_block_index, self.is_post_adapter)
        if self.bias_name is not None:
            self.bias_name = resolve_block_name(self.bias_name, block_index, adapter_block_index, self.is_post_adapter)

        lazy_load_file_path = get_lazy_load_file_path(self.lazy_load_file, self.weight_name)
        with safe_open(lazy_load_file_path, framework="pt", device="cpu") as lazy_load_file:
            weight_tensor = lazy_load_file.get_tensor(self.weight_name).t()
            self.pin_weight = self.pin_weight.copy_(weight_tensor)
            del weight_tensor

            if self.bias_name is not None:
                bias_tensor = lazy_load_file.get_tensor(self.bias_name)
                self.pin_bias.copy_(bias_tensor)
                del bias_tensor


class MMWeightQuantTemplate(MMWeightTemplate):
    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.weight_scale_name = self.weight_name.removesuffix(".weight") + ".weight_scale"
        self.load_func = None
        self.weight_need_transpose = True
        self.act_quant_func = None
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.infer_dtype = GET_DTYPE()
        self.bias_force_fp32 = False
        self.scale_force_fp32 = False
        self._update_base_attrs()

    def _update_base_attrs(self):
        self.base_attrs = [(self.weight_name, "weight", False), (self.weight_scale_name, "weight_scale", False)]
        if self.bias_name is not None:
            self.base_attrs.append((self.bias_name, "bias", False))

    # =========================
    # weight load functions
    # =========================
    def load(self, weight_dict):
        self.load_quantized(weight_dict)
        self.post_process()

    def post_process(self):
        if self.weight_need_transpose:
            if hasattr(self, "weight") and self.weight is not None:
                self.weight = self.weight.t()
            if hasattr(self, "pin_weight") and self.pin_weight is not None:
                self.pin_weight = self.pin_weight.t()
            if hasattr(self, "weight_cuda_buffer") and self.weight_cuda_buffer is not None:
                self.weight_cuda_buffer = self.weight_cuda_buffer.t()

        if hasattr(self, "bias") and self.bias is not None:
            if self.bias_force_fp32:
                self.bias = self.bias.to(torch.float32)
            else:
                self.bias = self.bias.to(self.infer_dtype)
        if hasattr(self, "pin_bias") and self.pin_bias is not None:
            if self.bias_force_fp32:
                self.pin_bias = self.pin_bias.to(torch.float32)
            else:
                self.pin_bias = self.pin_bias.to(self.infer_dtype)
        if self.bias_force_fp32 and hasattr(self, "bias_diff"):
            self.bias_diff = self.bias_diff.to(torch.float32)

        if hasattr(self, "weight_scale") and self.weight_scale is not None:
            self.weight_scale = self.weight_scale.to(torch.float32)
        if hasattr(self, "pin_weight_scale") and self.pin_weight_scale is not None:
            self.pin_weight_scale = self.pin_weight_scale.to(torch.float32)

    def load_quantized(self, weight_dict):
        if not self.create_cuda_buffer and not self.create_cpu_buffer and not self.lazy_load:
            device_tensors, pin_tensors = create_default_tensors(self.base_attrs, weight_dict)
            self.weight = device_tensors.get("weight")
            self.weight_scale = device_tensors.get("weight_scale")
            self.bias = device_tensors.get("bias")
            self.pin_weight = pin_tensors.get("weight")
            self.pin_weight_scale = pin_tensors.get("weight_scale")
            self.pin_bias = pin_tensors.get("bias")
        elif self.create_cuda_buffer:
            result = create_cuda_buffers(self.base_attrs, weight_dict, self.lazy_load, self.lazy_load_file, scale_force_fp32=self.scale_force_fp32)
            self.weight_cuda_buffer = result.get("weight")
            self.weight_scale_cuda_buffer = result.get("weight_scale")
            self.bias_cuda_buffer = result.get("bias")
        elif self.create_cpu_buffer:
            result = create_cpu_buffers(self.base_attrs, self.lazy_load_file, scale_force_fp32=self.scale_force_fp32)
            self.pin_weight = result.get("weight")
            self.pin_weight_scale = result.get("weight_scale")
            self.pin_bias = result.get("bias")
            self.weight = None
            self.weight_scale = None
            self.bias = None

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
            self.weight = weight_dict[self.weight_name].to(AI_DEVICE).to(torch.bfloat16)
            self.weight, self.weight_scale = scaled_mxfp4_quant(self.weight)
            self.weight, self.weight_scale = self.weight.to(device), self.weight_scale.to(device)
        else:
            self.load_quantized(weight_dict)

    def load_mxfp6(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            device = weight_dict[self.weight_name].device
            self.weight = weight_dict[self.weight_name].to(AI_DEVICE).to(torch.bfloat16)
            self.weight, self.weight_scale = scaled_mxfp6_quant(self.weight)
            self.weight, self.weight_scale = self.weight.to(device), self.weight_scale.to(device)
        else:
            self.load_quantized(weight_dict)

    def load_mxfp8(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            device = weight_dict[self.weight_name].device
            self.weight = weight_dict[self.weight_name].to(AI_DEVICE).to(torch.bfloat16)
            self.weight, self.weight_scale = scaled_mxfp8_quant(self.weight)
            self.weight, self.weight_scale = self.weight.to(device), self.weight_scale.to(device)
        else:
            self.load_quantized(weight_dict)

    def load_nvfp4(self, weight_dict):
        assert not self.config.get("weight_auto_quant", False)
        self.load_quantized(weight_dict)

    def load_fp8_perblock128_sym(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            self.weight = weight_dict[self.weight_name]
            self.weight, self.weight_scale = self.per_block_cast_to_fp8(self.weight)
        else:
            self.load_quantized(weight_dict)

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        if self.has_lora_branch or self.has_diff:
            self.load_lora_state_dict_from_disk(block_index)
        self.weight_name = resolve_block_name(self.weight_name, block_index, adapter_block_index, self.is_post_adapter)
        self.weight_scale_name = resolve_block_name(self.weight_scale_name, block_index, adapter_block_index, self.is_post_adapter)
        if self.bias_name is not None:
            self.bias_name = resolve_block_name(self.bias_name, block_index, adapter_block_index, self.is_post_adapter)

        lazy_load_file_path = get_lazy_load_file_path(self.lazy_load_file, self.weight_name)
        with safe_open(lazy_load_file_path, framework="pt", device="cpu") as lazy_load_file:
            if self.weight_need_transpose:
                weight_tensor = lazy_load_file.get_tensor(self.weight_name).t()
            else:
                weight_tensor = lazy_load_file.get_tensor(self.weight_name)

            self.pin_weight = self.pin_weight.copy_(weight_tensor)
            del weight_tensor

            weight_scale_tensor = lazy_load_file.get_tensor(self.weight_scale_name)
            self.pin_weight_scale = self.pin_weight_scale.copy_(weight_scale_tensor)
            del weight_scale_tensor

            if self.bias_name is not None:
                bias_tensor = lazy_load_file.get_tensor(self.bias_name)
                self.pin_bias.copy_(bias_tensor)
                del bias_tensor

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
        input_tensor_quant, input_tensor_scale = torchao_int8_quant(x)
        return input_tensor_quant, input_tensor_scale

    def act_quant_fp8_perchannel_sym_torchao(self, x):
        abs_max = x.abs().max(dim=-1, keepdim=True)[0]
        abs_max = torch.clamp(abs_max, min=1e-8)
        scale = abs_max / 448.0
        quantized = torch.clamp(x / scale, -448, 448).to(torch.float8_e4m3fn)
        return quantized, scale.float()

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


@MM_WEIGHT_REGISTER("fp8-vllm")
class MMWeightWfp8channelAfp8channeldynamicVllm(MMWeightQuantTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: vllm
    """

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.load_func = self.load_fp8_perchannel_sym
        self.act_quant_func = self.act_quant_fp8_perchannel_sym_vllm
        self.weight_need_transpose = True
        self.scale_force_fp32 = True

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
            self._get_actual_bias().to(self.infer_dtype),
        )
        if self.has_lora_branch:
            return output_tensor + self.apply_lora(input_tensor)
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

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.load_func = self.load_int8_perchannel_sym
        self.act_quant_func = self.act_quant_int8_perchannel_sym_vllm
        self.weight_need_transpose = True
        self.scale_force_fp32 = True

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
            self._get_actual_bias().to(self.infer_dtype),
        )
        if self.has_lora_branch:
            return output_tensor + self.apply_lora(input_tensor)
        return output_tensor


@MM_WEIGHT_REGISTER("mxfp4")
class MMWeightWmxfp4Amxfp4dynamic(MMWeightQuantTemplate):
    """
    Name: W-mxfp4-A-mxfp4-dynamic

    Quant MM:
        Weight: mxfp4
        Act: mxfp4
    """

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.load_func = self.load_mxfp4
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_mxfp4
        self.set_alpha()

    def set_alpha(self):
        self.alpha = torch.tensor(1.0, dtype=torch.float32)

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        self.alpha = self.alpha.to(self.weight.device)
        output_tensor = cutlass_scaled_mxfp4_mm(
            input_tensor_quant,
            self.weight,
            input_tensor_scale,
            self.weight_scale,
            alpha=self.alpha,
            bias=self._get_actual_bias(),
        )
        if self.has_lora_branch:
            return output_tensor + self.apply_lora(input_tensor)
        return output_tensor


@MM_WEIGHT_REGISTER("mxfp6-mxfp8")
class MMWeightWmxfp6Amxfp8dynamic(MMWeightQuantTemplate):
    """
    Name: W-mxfp6-A-nvfp8-dynamic

    Quant MM:
        Weight: mxfp6
        Act: mxfp8
    """

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.load_func = self.load_mxfp6
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_mxfp8
        self.set_alpha()

    def set_alpha(self):
        self.alpha = torch.tensor(1.0, dtype=torch.float32)

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        self.alpha = self.alpha.to(self.weight.device)
        output_tensor = cutlass_scaled_mxfp6_mxfp8_mm(
            input_tensor_quant,
            self.weight,
            input_tensor_scale,
            self.weight_scale,
            alpha=self.alpha,
            bias=self._get_actual_bias(),
        )
        if self.has_lora_branch:
            return output_tensor + self.apply_lora(input_tensor)
        return output_tensor


@MM_WEIGHT_REGISTER("mxfp8")
class MMWeightWmxfp8Amxfp8dynamic(MMWeightQuantTemplate):
    """
    Name: W-mxfp8-A-nvfp8-dynamic

    Quant MM:
        Weight: mxfp8
        Act: mxfp8
    """

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.load_func = self.load_mxfp8
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_mxfp8
        self.set_alpha()

    def set_alpha(self):
        self.alpha = torch.tensor(1.0, dtype=torch.float32)

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        self.alpha = self.alpha.to(self.weight.device)
        output_tensor = cutlass_scaled_mxfp8_mm(
            input_tensor_quant,
            self.weight,
            input_tensor_scale,
            self.weight_scale,
            alpha=self.alpha,
            bias=self._get_actual_bias(),
        )
        if self.has_lora_branch:
            return output_tensor + self.apply_lora(input_tensor)
        return output_tensor


@MM_WEIGHT_REGISTER("nvfp4")
class MMWeightWnvfp4Anvfp4dynamic(MMWeightQuantTemplate):
    """
    Name: W-nvfp4-A-nvfp4-dynamic

    Quant MM:
        Weight: nvfp4
        Act: nvfp4
    """

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.load_func = self.load_nvfp4
        self.input_absmax_name = self.weight_name.replace(".weight", ".input_absmax")
        self.weight_global_scale_name = self.weight_name + "_global_scale"
        self.input_global_scale_name = self.weight_name.replace(".weight", ".input_global_scale")
        self.alpha_name = self.weight_name.replace(".weight", ".alpha")
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_nvfp4

    def _update_base_attrs(self):
        super()._update_base_attrs()
        self.base_attrs.append((self.input_global_scale_name, "input_global_scale", False))
        self.base_attrs.append((self.alpha_name, "alpha", False))

    def _compute_input_global_scale_and_alpha(self, source, is_lazy=False):
        if is_lazy:
            if hasattr(source, "get_tensor") and source.get_tensor(self.input_absmax_name) is not None:
                input_absmax = source.get_tensor(self.input_absmax_name)
                input_global_scale = (2688.0 / input_absmax).to(torch.float32)
                weight_global_scale = source.get_tensor(self.weight_global_scale_name)
                alpha = 1.0 / (input_global_scale * weight_global_scale)
                return input_global_scale, alpha
        else:
            if self.input_absmax_name in source:
                input_absmax = source[self.input_absmax_name]
                input_global_scale = (2688.0 / input_absmax).to(torch.float32)
                weight_global_scale = source[self.weight_global_scale_name]
                alpha = 1.0 / (input_global_scale * weight_global_scale)
                return input_global_scale, alpha
        return None, None

    def _post_process_special_tensors(self, weight_dict, result=None, device_tensors=None, pin_tensors=None):
        if self.lazy_load:
            lazy_load_file_path = get_lazy_load_file_path(self.lazy_load_file, self.weight_name)
            with safe_open(lazy_load_file_path, framework="pt", device="cpu") as source:
                input_global_scale, alpha = self._compute_input_global_scale_and_alpha(source, is_lazy=True)
        else:
            source = weight_dict
            input_global_scale, alpha = self._compute_input_global_scale_and_alpha(source, is_lazy=False)

        if input_global_scale is not None and alpha is not None:
            if result is not None:
                if self.create_cuda_buffer:
                    result["input_global_scale"] = input_global_scale.to(AI_DEVICE)
                    result["alpha"] = alpha.to(AI_DEVICE)
                elif self.create_cpu_buffer:
                    result["input_global_scale"] = create_pin_tensor(input_global_scale)
                    result["alpha"] = create_pin_tensor(alpha)
            else:
                device = weight_dict[self.weight_name].device if self.weight_name in weight_dict else torch.device("cpu")
                if device.type == "cpu":
                    pin_tensors["input_global_scale"] = create_pin_tensor(input_global_scale)
                    pin_tensors["alpha"] = create_pin_tensor(alpha)
                else:
                    device_tensors["input_global_scale"] = input_global_scale.to(AI_DEVICE)
                    device_tensors["alpha"] = alpha.to(AI_DEVICE)

    def load_quantized(self, weight_dict):
        if not self.create_cuda_buffer and not self.create_cpu_buffer and not self.lazy_load:
            device_tensors, pin_tensors = create_default_tensors(self.base_attrs, weight_dict)
            self._post_process_special_tensors(weight_dict, device_tensors=device_tensors, pin_tensors=pin_tensors)
            self.weight = device_tensors.get("weight")
            self.weight_scale = device_tensors.get("weight_scale")
            self.bias = device_tensors.get("bias")
            self.input_global_scale = device_tensors.get("input_global_scale")
            self.alpha = device_tensors.get("alpha")

            self.pin_weight = pin_tensors.get("weight")
            self.pin_weight_scale = pin_tensors.get("weight_scale")
            self.pin_bias = pin_tensors.get("bias")
            self.pin_input_global_scale = pin_tensors.get("input_global_scale")
            self.pin_alpha = pin_tensors.get("alpha")
        elif self.create_cuda_buffer:
            result = create_cuda_buffers(self.base_attrs, weight_dict, self.lazy_load, self.lazy_load_file)
            self._post_process_special_tensors(weight_dict, result=result)
            self.weight_cuda_buffer = result.get("weight")
            self.weight_scale_cuda_buffer = result.get("weight_scale")
            self.bias_cuda_buffer = result.get("bias")
            self.input_global_scale_cuda_buffer = result.get("input_global_scale")
            self.alpha_cuda_buffer = result.get("alpha")
        elif self.create_cpu_buffer:
            result = create_cpu_buffers(self.base_attrs, self.lazy_load_file)
            self._post_process_special_tensors(weight_dict, result=result)
            self.pin_weight = result.get("weight")
            self.pin_weight_scale = result.get("weight_scale")
            self.pin_bias = result.get("bias")
            self.pin_input_global_scale = result.get("input_global_scale")
            self.pin_alpha = result.get("alpha")

            self.weight = None
            self.weight_scale = None
            self.bias = None

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = cutlass_scaled_nvfp4_mm(
            input_tensor_quant,
            self.weight,
            input_tensor_scale,
            self.weight_scale,
            alpha=self.alpha,
            bias=self._get_actual_bias(),
        )
        if self.has_lora_branch:
            return output_tensor + self.apply_lora(input_tensor)
        return output_tensor

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.weight_name] = self.pin_weight if hasattr(self, "pin_weight") else self.weight
        if self.bias_name is not None:
            destination[self.bias_name] = self.pin_bias if hasattr(self, "pin_bias") else self.bias
        destination[self.weight_scale_name] = self.pin_weight_scale if hasattr(self, "pin_weight_scale") else self.weight_scale

        destination[self.input_global_scale_name] = self.pin_input_global_scale if hasattr(self, "pin_input_global_scale") else self.input_global_scale
        destination[self.alpha_name] = self.pin_alpha if hasattr(self, "pin_alpha") else self.alpha

        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        if self.is_post_adapter:
            weight_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.weight_name, count=1)
            weight_scale_name = re.sub(
                r"\.\d+",
                lambda m: f".{adapter_block_index}",
                self.weight_scale_name,
                count=1,
            )
            input_global_scale_name = re.sub(
                r"\.\d+",
                lambda m: f".{adapter_block_index}",
                self.input_global_scale_name,
                count=1,
            )
            alpha_name = re.sub(
                r"\.\d+",
                lambda m: f".{adapter_block_index}",
                self.alpha_name,
                count=1,
            )
        else:
            weight_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_name, count=1)
            weight_scale_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_scale_name, count=1)
            input_global_scale_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.input_global_scale_name, count=1)
            alpha_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.alpha_name, count=1)

        if weight_name not in destination:
            self.weight = None
            return

        self.weight = self.weight_cuda_buffer.copy_(destination[weight_name], non_blocking=True)
        self.weight_scale = self.weight_scale_cuda_buffer.copy_(destination[weight_scale_name], non_blocking=True)
        self.input_global_scale = self.input_global_scale_cuda_buffer.copy_(destination[input_global_scale_name], non_blocking=True)
        self.alpha = self.alpha_cuda_buffer.copy_(destination[alpha_name], non_blocking=True)
        if self.bias_name is not None:
            bias_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.bias_name, count=1)
            self.bias = self.bias_cuda_buffer.copy_(destination[bias_name], non_blocking=True)
        else:
            self.bias = None

    # def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
    #     if self.is_post_adapter:
    #         self.weight_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.weight_name, count=1)
    #         self.weight_scale_name = re.sub(
    #             r"\.\d+",
    #             lambda m: f".{adapter_block_index}",
    #             self.weight_scale_name,
    #             count=1,
    #         )
    #         self.input_global_scale_name = re.sub(
    #             r"\.\d+",
    #             lambda m: f".{adapter_block_index}",
    #             self.input_global_scale_name,
    #             count=1,
    #         )
    #         self.alpha_name = re.sub(
    #             r"\.\d+",
    #             lambda m: f".{adapter_block_index}",
    #             self.alpha_name,
    #             count=1,
    #         )
    #     else:
    #         self.weight_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_name, count=1)
    #         self.weight_scale_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_scale_name, count=1)
    #         self.input_global_scale_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.input_global_scale_name, count=1)
    #         self.alpha_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.alpha_name, count=1)

    #     if self.bias_name is not None:
    #         if self.is_post_adapter:
    #             assert adapter_block_index is not None
    #             self.bias_name = re.sub(
    #                 r"\.\d+",
    #                 lambda m: f".{adapter_block_index}",
    #                 self.bias_name,
    #                 count=1,
    #             )
    #         else:
    #             self.bias_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.bias_name, count=1)
    #     if Path(self.lazy_load_file).is_file():
    #         lazy_load_file_path = self.lazy_load_file
    #     else:
    #         lazy_load_file_path = os.path.join(self.lazy_load_file, f"block_{block_index}.safetensors")
    #     with safe_open(lazy_load_file_path, framework="pt", device="cpu") as lazy_load_file:
    #         if self.weight_need_transpose:
    #             weight_tensor = lazy_load_file.get_tensor(self.weight_name).t()
    #         else:
    #             weight_tensor = lazy_load_file.get_tensor(self.weight_name)

    #         self.pin_weight = self.pin_weight.copy_(weight_tensor)
    #         del weight_tensor

    #         weight_scale_tensor = lazy_load_file.get_tensor(self.weight_scale_name)
    #         self.pin_weight_scale = self.pin_weight_scale.copy_(weight_scale_tensor)
    #         del weight_scale_tensor

    #         input_global_scale_tensor = lazy_load_file.get_tensor(self.input_global_scale_name)
    #         self.pin_input_global_scale = self.pin_input_global_scale.copy_(input_global_scale_tensor)
    #         del input_global_scale_tensor

    #         alpha_tensor = lazy_load_file.get_tensor(self.alpha_name)
    #         self.pin_alpha = self.pin_alpha.copy_(alpha_tensor)
    #         del alpha_tensor

    #         if self.bias_name is not None:
    #             bias_tensor = lazy_load_file.get_tensor(self.bias_name)
    #             self.pin_bias.copy_(bias_tensor)
    #             del bias_tensor


@MM_WEIGHT_REGISTER("Calib")
class MMCalibNvfp4(MMWeight):
    """
    Name: calib

    Calib:
        absmax: torch.max(torch.abs(input_tensor))
    """

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
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
        if hasattr(self, "bias") and self.bias is not None:
            return torch.addmm(self.bias, input_tensor, self.weight, out=output_tensor)
        return torch.mm(input_tensor, self.weight, out=output_tensor)


@MM_WEIGHT_REGISTER("fp8-q8f")
class MMWeightWfp8channelAfp8channeldynamicQ8F(MMWeightQuantTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Q8F

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: Q8F
    """

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.load_func = self.load_fp8_perchannel_sym
        self.weight_need_transpose = False
        self.bias_force_fp32 = True
        if ops is not None:
            self.act_quant_func = self.act_quant_fp8_perchannel_sym_vllm
        else:
            self.act_quant_func = fp8_quantize_triton

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = fp8_linear(
            input_tensor_quant,
            self.weight,
            self._get_actual_bias(),
            input_tensor_scale.float(),
            self.weight_scale,
            out_dtype=self.infer_dtype,
        )
        if self.has_lora_branch:
            return output_tensor.squeeze(0) + self.apply_lora(input_tensor) if len(output_tensor.shape) == 3 else output_tensor + +self.apply_lora(input_tensor)
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

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.load_func = self.load_int8_perchannel_sym
        self.weight_need_transpose = False
        self.bias_force_fp32 = True
        if ops is not None:
            self.act_quant_func = self.act_quant_int8_perchannel_sym_vllm
        else:
            self.act_quant_func = int8_quantize_triton

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = q8_linear(
            input_tensor_quant,
            self.weight,
            self._get_actual_bias(),
            input_tensor_scale.float(),
            self.weight_scale,
            fuse_gelu=False,
            out_dtype=self.infer_dtype,
        )
        if self.has_lora_branch:
            return output_tensor.squeeze(0) + self.apply_lora(input_tensor) if len(output_tensor.shape) == 3 else output_tensor + +self.apply_lora(input_tensor)
        return output_tensor.squeeze(0) if len(output_tensor.shape) == 3 else output_tensor


@MM_WEIGHT_REGISTER("fp8-triton")
class MMWeightWfp8channelAfp8channeldynamicTriton(MMWeightQuantTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-triton

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: triton
    """

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.load_func = self.load_fp8_perchannel_sym
        self.act_quant_func = fp8_quantize_triton
        self.weight_need_transpose = False
        self.bias_force_fp32 = True

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        if self.bias is not None:
            output_tensor = fp8_gemm_bias_triton(
                input_tensor_quant,
                self.weight,
                self._get_actual_bias(),
                input_tensor_scale,
                self.weight_scale,
                output_dtype=self.infer_dtype,
            )
        else:
            output_tensor = fp8_gemm_triton(
                input_tensor_quant,
                self.weight,
                input_tensor_scale,
                self.weight_scale,
                output_dtype=self.infer_dtype,
            )
        if self.has_lora_branch:
            return output_tensor.squeeze(0) + self.apply_lora(input_tensor) if len(output_tensor.shape) == 3 else output_tensor + +self.apply_lora(input_tensor)
        return output_tensor.squeeze(0) if len(output_tensor.shape) == 3 else output_tensor


@MM_WEIGHT_REGISTER("int8-triton")
class MMWeightWint8channelAint8channeldynamicTriton(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-triton

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: triton
    """

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.load_func = self.load_int8_perchannel_sym
        self.act_quant_func = int8_quantize_triton
        self.weight_need_transpose = False
        self.bias_force_fp32 = True

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        if self.bias is not None:
            output_tensor = int8_gemm_bias_triton(
                input_tensor_quant,
                self.weight,
                self._get_actual_bias(),
                input_tensor_scale,
                self.weight_scale,
                output_dtype=self.infer_dtype,
            )
        else:
            output_tensor = int8_gemm_triton(
                input_tensor_quant,
                self.weight,
                input_tensor_scale,
                self.weight_scale,
                output_dtype=self.infer_dtype,
            )
        if self.has_lora_branch:
            return output_tensor.squeeze(0) + self.apply_lora(input_tensor) if len(output_tensor.shape) == 3 else output_tensor + +self.apply_lora(input_tensor)

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

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
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
            output_tensor.add_(self._get_actual_bias())
        if self.has_lora_branch:
            return output_tensor + self.apply_lora(input_tensor)
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

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.weight_need_transpose = True
        self.scale_force_fp32 = True
        self.load_func = self.load_fp8_perchannel_sym
        self.act_quant_func = self.act_quant_fp8_perchannel_sym_sgl

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = sgl_kernel.fp8_scaled_mm(
            input_tensor_quant,
            self.weight,
            input_tensor_scale,
            self.weight_scale,
            self.infer_dtype,
            self._get_actual_bias().to(self.infer_dtype),
        )
        if self.has_lora_branch:
            return output_tensor + self.apply_lora(input_tensor)
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

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.load_func = self.load_int8_perchannel_sym
        self.act_quant_func = self.act_quant_int8_perchannel_sym_vllm
        self.weight_need_transpose = True
        self.scale_force_fp32 = True

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
            self._get_actual_bias().to(self.infer_dtype),
        )
        if self.has_lora_branch:
            return output_tensor + self.apply_lora(input_tensor)
        return output_tensor


@MM_WEIGHT_REGISTER("fp8-torchao")
class MMWeightWfp8channelAfp8channeldynamicTorchao(MMWeightQuantTemplate):
    """
    Name: W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Torchao

    Quant MM:
        Weight: fp8 perchannel sym
        Act: fp8 perchannel dynamic sym
        Kernel: Torchao
    """

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.load_func = self.load_fp8_perchannel_sym
        self.act_quant_func = self.act_quant_fp8_perchannel_sym_torchao
        self.weight_need_transpose = True
        self.scale_force_fp32 = True

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_fp8_perchannel_sym_torchao(input_tensor)
        output_tensor = torch._scaled_mm(
            input_tensor_quant,
            self.weight,
            scale_a=input_tensor_scale.float(),
            scale_b=self.weight_scale.t(),
            bias=self._get_actual_bias().to(self.infer_dtype),
            out_dtype=self.infer_dtype,
            use_fast_accum=True,
        )
        if self.has_lora_branch:
            return output_tensor + self.apply_lora(input_tensor)
        return output_tensor


@MM_WEIGHT_REGISTER("int8-torchao")
class MMWeightWint8channelAint8channeldynamicTorchao(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Torchao

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: Torchao
    """

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        self.load_func = self.load_int8_perchannel_sym
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_int8_perchannel_sym_torchao

    def apply(self, input_tensor):
        input_tensor = input_tensor
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = torchao_int8_gemm(
            input_tensor_quant,
            input_tensor_scale,
            self.weight,
            self.weight_scale.t().float(),
            output_dtype=self.infer_dtype,
        )
        if self.bias is not None:
            output_tensor.add_(self._get_actual_bias())

        if self.has_lora_branch:
            return output_tensor + self.apply_lora(input_tensor)
        return output_tensor


class MMWeightGGUFTemplate(MMWeightTemplate):
    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )

    def load(self, weight_dict):
        if not self.lazy_load:
            assert not self.create_cuda_buffer, "GGUF Unsupported offload block"
            self.weight = weight_dict[self.weight_name]

            weight_shape = self.weight.shape
            weight_dtype = self.weight.dtype

            if isinstance(self.weight, GGMLTensor):
                self.pin_weight = GGMLTensor.empty_pinned(
                    weight_shape,
                    orig_shape=self.weight.orig_shape,
                    dtype=weight_dtype,
                    gguf_type=self.weight.gguf_type,
                )
                self.pin_weight.copy_from(self.weight)
            else:
                self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
                self.pin_weight.copy_(weight_dict[self.weight_name])

            if self.bias_name is not None:
                self.bias = weight_dict[self.bias_name]
                if isinstance(self.bias, GGMLTensor):
                    self.pin_bias = GGMLTensor.empty_pinned(
                        self.bias.shape,
                        orig_shape=self.bias.orig_shape,
                        dtype=self.bias.dtype,
                        gguf_type=self.bias.gguf_type,
                    )
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
                bias_name = re.sub(
                    r"\.\d+",
                    lambda m: f".{adapter_block_index}",
                    self.bias_name,
                    count=1,
                )
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

        weight = gguf_dequantize_tensor(tensor, dtype)
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
        if self.has_lora_branch:
            return torch.nn.functional.linear(input_tensor, weight, self._get_actual_bias(bias)) + self.apply_lora(input_tensor)
        return torch.nn.functional.linear(input_tensor, weight, self._get_actual_bias(bias))


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

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
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
        output_tensor = torch.empty(
            input_tensor.shape[:-1] + (self.weight_scale.shape[1],),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        marlin_cuda_quant.mul(
            input_tensor,
            self.weight,
            output_tensor,
            self.weight_scale.half(),
            self.workspace,
            -1,
            -1,
            -1,
            -1,
        )
        if hasattr(self, "bias") and self.bias is not None:
            output_tensor.add_(self._get_actual_bias())
        if self.has_lora_branch:
            return output_tensor + self.apply_lora(input_tensor)
        return output_tensor
