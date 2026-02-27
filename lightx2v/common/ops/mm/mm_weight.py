import re
from abc import ABCMeta, abstractmethod

import torch
import torch.distributed as dist
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
    from vllm import _custom_ops as vllm_ops
except ImportError:
    vllm_ops = None


try:
    from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8 as sglang_int8_act_quant
except ImportError:
    sglang_int8_act_quant = None

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

import torch.distributed as dist


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
            if not hasattr(self, "bias") or self.bias is None:
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

    def update_lora(self, weight_dict, lora_strength=1):
        if not self.lazy_load or self.create_cuda_buffer or self.create_cpu_buffer:
            if self.lora_down_name in weight_dict:
                self.has_lora_branch = True
                self.lora_down.copy_(weight_dict[self.lora_down_name])
                self.lora_up.copy_(weight_dict[self.lora_up_name])
                self.lora_strength = lora_strength
                if self.lora_alpha_name in weight_dict:
                    self.lora_alpha.copy_(weight_dict[self.lora_alpha_name])
                    self.lora_scale.copy_(self.lora_alpha / self.lora_down.shape[0])
                else:
                    self.lora_scale = torch.tensor(1.0, device=AI_DEVICE)
                logger.debug(f"Update LoRA to {self.weight_name}")

    def remove_lora(self):
        if hasattr(self, "lora_down"):
            del self.lora_down
        if hasattr(self, "lora_up"):
            del self.lora_up
        if hasattr(self, "lora_alpha"):
            del self.lora_alpha
        if hasattr(self, "lora_scale"):
            del self.lora_scale
        self.has_lora_branch = False
        logger.debug(f"Remove LoRA from {self.weight_name}")

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
        bias_name=None,
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
        if self.scale_force_fp32:
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
            result = create_cuda_buffers(self.base_attrs, weight_dict, self.lazy_load, self.lazy_load_file, scale_force_fp32=self.scale_force_fp32, bias_force_fp32=self.bias_force_fp32)
            self.weight_cuda_buffer = result.get("weight")
            self.weight_scale_cuda_buffer = result.get("weight_scale")
            self.bias_cuda_buffer = result.get("bias")
        elif self.create_cpu_buffer:
            result = create_cpu_buffers(self.base_attrs, self.lazy_load_file, scale_force_fp32=self.scale_force_fp32, bias_force_fp32=self.bias_force_fp32)
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
        if self.scale_force_fp32:
            input_tensor_scale = input_tensor_scale.to(torch.float32)
        return input_tensor_quant, input_tensor_scale

    def act_quant_fp8_perchannel_sym_torchao(self, x):
        abs_max = x.abs().max(dim=-1, keepdim=True)[0]
        abs_max = torch.clamp(abs_max, min=1e-8)
        scale = abs_max / 448.0
        quantized = torch.clamp(x / scale, -448, 448).to(torch.float8_e4m3fn)
        return quantized, scale.float()

    def act_quant_fp8_perchannel_sym_vllm(self, x):
        input_tensor_quant, input_tensor_scale = vllm_ops.scaled_fp8_quant(x, None, scale_ub=None, use_per_token_if_dynamic=True)
        return input_tensor_quant, input_tensor_scale

    def act_quant_fp8_perchannel_sym_sgl(self, x):
        m, k = x.shape
        input_tensor_quant = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda", requires_grad=False)
        input_tensor_scale = torch.empty((m, 1), dtype=torch.float32, device="cuda", requires_grad=False)
        sgl_kernel.sgl_per_token_quant_fp8(x, input_tensor_quant, input_tensor_scale)
        return input_tensor_quant, input_tensor_scale

    def act_quant_int8_perchannel_sym_vllm(self, x):
        input_tensor_quant, input_tensor_scale, _ = vllm_ops.scaled_int8_quant(x, scale=None, azp=None, symmetric=True)
        return input_tensor_quant, input_tensor_scale

    def act_quant_nvfp4(self, x):
        # input_global_scale= x.abs().max()
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
            self._get_actual_bias(),
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
            self._get_actual_bias(),
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
            lora_prefix=lora_prefix,
            lora_path=lora_path,
        )
        self.load_func = self.load_nvfp4
        self.input_absmax_name = self.weight_name.replace(".weight", ".input_absmax")
        self.weight_global_scale_name = self.weight_name + "_global_scale"
        self.input_global_scale_name = self.weight_name.replace(".weight", ".input_global_scale")
        self.alpha_name = self.weight_name.replace(".weight", ".alpha")
        self.act_quant_func = self.act_quant_nvfp4
        self.weight_need_transpose = False

    def load_quantized(self, weight_dict):
        if self.create_cuda_buffer:
            self._load_cuda_buffers(weight_dict)
        elif self.create_cpu_buffer:
            self._load_cpu_pin_buffers()
        else:
            self._load_default_tensors(weight_dict)

    def _load_cuda_buffers(self, weight_dict):
        if self.lazy_load:
            if Path(self.lazy_load_file).is_file():
                lazy_load_file_path = self.lazy_load_file
            else:
                lazy_load_file_path = os.path.join(
                    self.lazy_load_file,
                    f"block_{self.weight_name.split('.')[1]}.safetensors",
                )
            with safe_open(lazy_load_file_path, framework="pt", device="cpu") as source:
                (
                    self.weight_cuda_buffer,
                    self.weight_scale_cuda_buffer,
                    self.input_global_scale_cuda_buffer,
                    self.alpha_cuda_buffer,
                ) = self._get_cuda_tensor_pair(source, self.lazy_load)
                self.bias_cuda_buffer = self._get_cuda_bias_tensor(source, self.lazy_load)
        else:
            source = weight_dict
            (
                self.weight_cuda_buffer,
                self.weight_scale_cuda_buffer,
                self.input_global_scale_cuda_buffer,
                self.alpha_cuda_buffer,
            ) = self._get_cuda_tensor_pair(source, self.lazy_load)
            self.bias_cuda_buffer = self._get_cuda_bias_tensor(source, self.lazy_load)

    def _get_cuda_tensor_pair(self, source, is_lazy):
        if is_lazy:
            if self.input_absmax_name in source.keys():
                input_absmax = source.get_tensor(self.input_absmax_name)
                input_global_scale = (2688.0 / input_absmax).to(torch.float32).to(AI_DEVICE)
                weight_global_scale = source.get_tensor(self.weight_global_scale_name).to(AI_DEVICE)
                alpha = 1.0 / (input_global_scale * weight_global_scale)
            else:
                input_global_scale = source.get_tensor(self.input_global_scale_name).to(torch.float32).to(AI_DEVICE)
                alpha = source.get_tensor(self.alpha_name).to(torch.float32).to(AI_DEVICE)
            weight = source.get_tensor(self.weight_name).to(AI_DEVICE)
            scale = source.get_tensor(self.weight_scale_name).to(AI_DEVICE)
        else:
            if self.input_absmax_name in source:
                input_absmax = source[self.input_absmax_name]
                input_global_scale = (2688.0 / input_absmax).to(torch.float32).to(AI_DEVICE)
                weight_global_scale = source[self.weight_global_scale_name].to(AI_DEVICE)
                alpha = 1.0 / (input_global_scale * weight_global_scale)
            else:
                input_global_scale = source[self.input_global_scale_name].to(torch.float32).to(AI_DEVICE)
                alpha = source[self.alpha_name].to(torch.float32).to(AI_DEVICE)

            weight = source[self.weight_name].to(AI_DEVICE)
            scale = source[self.weight_scale_name].to(AI_DEVICE)
        return weight, scale, input_global_scale, alpha

    def _get_cuda_bias_tensor(self, source, is_lazy):
        if self.bias_name is None:
            return None
        if is_lazy:
            bias = source.get_tensor(self.bias_name)
            dtype = self.infer_dtype
        else:
            bias = source[self.bias_name]
            dtype = bias.dtype
        if self.bias_force_fp32:
            bias = bias.to(torch.float32)
        else:
            bias = bias.to(dtype)
        return bias.to(AI_DEVICE)

    def _load_cpu_pin_buffers(self):
        (
            self.pin_weight,
            self.pin_weight_scale,
            self.pin_input_global_scale,
            self.pin_alpha,
        ) = self._get_cpu_pin_tensor_pair(self.lazy_load_file, is_lazy=True)
        self.pin_bias = self._get_cpu_pin_bias_tensor(self.lazy_load_file, is_lazy=True)
        self.bias = None

    def _get_cpu_pin_tensor_pair(self, source, is_lazy):
        if is_lazy:
            if Path(self.lazy_load_file).is_file():
                lazy_load_file_path = self.lazy_load_file
            else:
                lazy_load_file_path = os.path.join(
                    self.lazy_load_file,
                    f"block_{self.weight_name.split('.')[1]}.safetensors",
                )
            with safe_open(lazy_load_file_path, framework="pt", device="cpu") as source:
                weight_tensor = source.get_tensor(self.weight_name)
                scale_tensor = source.get_tensor(self.weight_scale_name)
                if self.input_absmax_name in source.keys():
                    input_absmax = source.get_tensor(self.input_absmax_name)
                    input_global_scale = (2688.0 / input_absmax).to(torch.float32)
                    weight_global_scale = source.get_tensor(self.weight_global_scale_name)
                    alpha = 1.0 / (input_global_scale * weight_global_scale)
                else:
                    input_global_scale = source.get_tensor(self.input_global_scale_name).to(torch.float32)
                    alpha = source.get_tensor(self.alpha_name).to(torch.float32)
                pin_weight = self._create_pin_tensor(weight_tensor)
                pin_scale = self._create_pin_tensor(scale_tensor)
                pin_input_global_scale = self._create_pin_tensor(input_global_scale)
                pin_alpha = self._create_pin_tensor(alpha)
        else:
            weight_tensor = source[self.weight_name]
            scale_tensor = source[self.weight_scale_name]
            if self.input_absmax_name in source:
                input_absmax = source[self.input_absmax_name]
                input_global_scale = (2688.0 / input_absmax).to(torch.float32)
                weight_global_scale = source[self.weight_global_scale_name]
                alpha = 1.0 / (input_global_scale * weight_global_scale)
            else:
                input_global_scale = source[self.input_global_scale_name].to(torch.float32)
                alpha = source[self.alpha_name].to(torch.float32)
            pin_weight = self._create_pin_tensor(weight_tensor)
            pin_scale = self._create_pin_tensor(scale_tensor)
            pin_input_global_scale = self._create_pin_tensor(input_global_scale)
            pin_alpha = self._create_pin_tensor(alpha)

        return pin_weight, pin_scale, pin_input_global_scale, pin_alpha

    def _get_cpu_pin_bias_tensor(self, source, is_lazy):
        if self.bias_name is None:
            return None
        if is_lazy:
            if Path(self.lazy_load_file).is_file():
                lazy_load_file_path = self.lazy_load_file
            else:
                lazy_load_file_path = os.path.join(
                    self.lazy_load_file,
                    f"block_{self.weight_name.split('.')[1]}.safetensors",
                )
            with safe_open(lazy_load_file_path, framework="pt", device="cpu") as source:
                bias_tensor = source.get_tensor(self.bias_name)
                if not self.bias_force_fp32:
                    bias_tensor = bias_tensor.to(self.infer_dtype)
                if self.bias_force_fp32:
                    bias_tensor = bias_tensor.to(torch.float32)
                return self._create_pin_tensor(bias_tensor)
        else:
            bias_tensor = source[self.bias_name]
            if self.bias_force_fp32:
                bias_tensor = bias_tensor.to(torch.float32)
            return self._create_pin_tensor(bias_tensor)

    def _create_pin_tensor(self, tensor, dtype=None):
        dtype = dtype or tensor.dtype
        pin_tensor = torch.empty(tensor.shape, pin_memory=True, dtype=dtype)
        pin_tensor.copy_(tensor)
        del tensor
        return pin_tensor

    def _load_default_tensors(self, weight_dict):
        if not self.lazy_load:
            (
                self.weight,
                self.weight_scale,
                self.input_global_scale,
                self.alpha,
                self.pin_weight,
                self.pin_weight_scale,
                self.pin_input_global_scale,
                self.pin_alpha,
            ) = self._get_device_tensor_pair(weight_dict)
            self._load_default_bias(weight_dict)
        else:
            self.bias = None
            self.pin_bias = None

    def _get_device_tensor_pair(self, source):
        device = source[self.weight_name].device
        if device.type == "cpu":
            pin_weight, pin_scale, pin_input_global_scale, pin_alpha = self._get_cpu_pin_tensor_pair(source, is_lazy=False)
            return (
                None,
                None,
                None,
                None,
                pin_weight,
                pin_scale,
                pin_input_global_scale,
                pin_alpha,
            )
        else:
            if self.input_absmax_name in source:
                input_absmax = source[self.input_absmax_name]
                input_global_scale = (2688.0 / input_absmax).to(torch.float32)
                weight_global_scale = source[self.weight_global_scale_name]
                alpha = 1.0 / (input_global_scale * weight_global_scale)
            else:
                input_global_scale = source[self.input_global_scale_name].to(torch.float32).to(AI_DEVICE)
                alpha = source[self.alpha_name].to(torch.float32).to(AI_DEVICE)
            return (
                source[self.weight_name],
                source[self.weight_scale_name],
                input_global_scale,
                alpha,
                None,
                None,
                None,
                None,
            )

    def _load_default_bias(self, source):
        if self.bias_name is None:
            self.bias = None
            self.pin_bias = None
            self.bias_cuda_buffer = None
            return

        if self.create_cuda_buffer:
            self.bias_cuda_buffer = self._get_cuda_bias_tensor(source, is_lazy=False)
            self.bias = None
            self.pin_bias = None
        else:
            bias_tensor = source[self.bias_name].float() if self.bias_force_fp32 else source[self.bias_name]
            device = bias_tensor.device
            if device.type == "cpu":
                self.pin_bias = self._get_cpu_pin_bias_tensor(source, is_lazy=False)
                self.bias = None
            else:
                self.bias = bias_tensor
                self.pin_bias = None

    def apply(self, input_tensor):
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = cutlass_scaled_nvfp4_mm(
            input_tensor_quant,
            self.weight,
            input_tensor_scale,
            self.weight_scale,
            alpha=self.alpha,
            bias=self.bias,
        )
        return output_tensor

    def to_cuda(self, non_blocking=False):
        self.weight = self.pin_weight.to(AI_DEVICE, non_blocking=non_blocking)
        if hasattr(self, "pin_weight_scale"):
            self.weight_scale = self.pin_weight_scale.to(AI_DEVICE, non_blocking=non_blocking)
            self.input_global_scale = self.pin_input_global_scale.to(AI_DEVICE, non_blocking=non_blocking)
            self.alpha = self.pin_alpha.to(AI_DEVICE, non_blocking=non_blocking)
        if hasattr(self, "pin_bias") and self.pin_bias is not None:
            self.bias = self.pin_bias.to(AI_DEVICE, non_blocking=non_blocking)

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
        weight_name = resolve_block_name(self.weight_name, block_index, adapter_block_index, self.is_post_adapter)
        weight_scale_name = resolve_block_name(self.weight_scale_name, block_index, adapter_block_index, self.is_post_adapter)
        input_global_scale_name = resolve_block_name(self.input_global_scale_name, block_index, adapter_block_index, self.is_post_adapter)
        alpha_name = resolve_block_name(self.alpha_name, block_index, adapter_block_index, self.is_post_adapter)

        if weight_name not in destination:
            self.weight = None
            return

        self.weight = self.weight_cuda_buffer.copy_(destination[weight_name], non_blocking=True)
        self.weight_scale = self.weight_scale_cuda_buffer.copy_(destination[weight_scale_name], non_blocking=True)
        self.input_global_scale = self.input_global_scale_cuda_buffer.copy_(destination[input_global_scale_name], non_blocking=True)
        self.alpha = self.alpha_cuda_buffer.copy_(destination[alpha_name], non_blocking=True)
        if self.bias_name is not None:
            bias_name = resolve_block_name(self.bias_name, block_index, adapter_block_index, self.is_post_adapter)
            self.bias = self.bias_cuda_buffer.copy_(destination[bias_name], non_blocking=True)
        else:
            self.bias = None

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        self.weight_name = resolve_block_name(self.weight_name, block_index, adapter_block_index, self.is_post_adapter)
        self.weight_scale_name = resolve_block_name(self.weight_scale_name, block_index, adapter_block_index, self.is_post_adapter)
        self.input_global_scale_name = resolve_block_name(self.input_global_scale_name, block_index, adapter_block_index, self.is_post_adapter)
        self.alpha_name = resolve_block_name(self.alpha_name, block_index, adapter_block_index, self.is_post_adapter)

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
        self.scale_force_fp32 = True
        if vllm_ops is not None:
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
            return output_tensor.squeeze(0) + self.apply_lora(input_tensor) if len(output_tensor.shape) == 3 else output_tensor + self.apply_lora(input_tensor)
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
        self.scale_force_fp32 = True
        if vllm_ops is not None:
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
            self._get_actual_bias(),
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
        Kernel: quant-mm using Sgl-kernel, act dynamic quant using sglang and fallback to whatever quant backend available with this priority: vllm > torchao > triton
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
        # Priority sglang > vllm > toarchao > triton
        if sglang_int8_act_quant:
            self.act_quant_func = sglang_int8_act_quant
        elif vllm_ops:
            self.act_quant_func = self.act_quant_int8_perchannel_sym_vllm
        elif torchao_int8_quant:
            self.act_quant_func = self.act_quant_int8_perchannel_sym_torchao
        else:
            self.act_quant_func = int8_quantize_triton
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
            self._get_actual_bias(),
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
            bias=self._get_actual_bias(),
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


@MM_WEIGHT_REGISTER("fp8-pertensor")
class MMWeightWfp8tensorAfp8tensordynamic(MMWeightQuantTemplate):
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
        self.load_func = self.load_fp8_pertensor_sym
        self.act_quant_func = self.act_quant_fp8_pertensor_sym
        self.weight_need_transpose = True
        self.scale_force_fp32 = True

    def _update_base_attrs(self):
        super()._update_base_attrs()
        self.input_scale_name = self.weight_name.removesuffix(".weight") + ".input_scale"
        self.base_attrs.append((self.input_scale_name, "input_scale", False))

    def load_quantized(self, weight_dict):
        super().load_quantized(weight_dict)
        if not self.create_cuda_buffer and not self.create_cpu_buffer and not self.lazy_load:
            device_tensors, pin_tensors = create_default_tensors(self.base_attrs, weight_dict)
            self.input_scale = device_tensors.get("input_scale")
            self.pin_input_scale = pin_tensors.get("input_scale")
        elif self.create_cuda_buffer:
            result = create_cuda_buffers(self.base_attrs, weight_dict, self.lazy_load, self.lazy_load_file, scale_force_fp32=self.scale_force_fp32, bias_force_fp32=self.bias_force_fp32)
            self.input_scale_cuda_buffer = result.get("input_scale")
        elif self.create_cpu_buffer:
            result = create_cpu_buffers(self.base_attrs, self.lazy_load_file, scale_force_fp32=self.scale_force_fp32, bias_force_fp32=self.bias_force_fp32)
            self.pin_input_scale = result.get("input_scale")
            self.input_scale = None

    def post_process(self):
        super().post_process()
        if self.scale_force_fp32:
            if hasattr(self, "input_scale") and self.input_scale is not None:
                self.input_scale = self.input_scale.to(torch.float32)
            if hasattr(self, "pin_input_scale") and self.pin_input_scale is not None:
                self.pin_input_scale = self.pin_input_scale.to(torch.float32)

    def load_fp8_pertensor_sym(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            raise NotImplementedError
        else:
            self.load_quantized(weight_dict)

    def act_quant_fp8_pertensor_sym(self, x):
        quantized = torch.clamp(x / self.input_scale, -448, 448).to(torch.float8_e4m3fn)
        return quantized

    def apply(self, input_tensor):
        shape = (input_tensor.shape[0], self.weight.shape[1])
        dtype = input_tensor.dtype
        device = input_tensor.device
        output_tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)
        input_tensor_quant = self.act_quant_func(input_tensor)
        output_tensor = torch._scaled_mm(
            input_tensor_quant,
            self.weight,
            scale_a=self.input_scale,
            scale_b=self.weight_scale.reshape(1),
            bias=self._get_actual_bias(),
            out_dtype=dtype,
            use_fast_accum=True,
        )
        if self.has_lora_branch:
            return output_tensor + self.apply_lora(input_tensor)
        return output_tensor


@MM_WEIGHT_REGISTER("TensorParallel")
class MMWeightTP(MMWeightTemplate):
    """
    Tensor Parallel wrapper for any MMWeight type.

    This is a generic wrapper that can wrap any MMWeight implementation (Default, fp8, int8, etc.)
    and add tensor parallelism support by:
    1. Handling weight splitting in load() method
    2. Adding all-reduce for row-wise split in apply() method

    Supports column-wise and row-wise weight splitting:
    - Column split: weight [in_dim, out_dim] -> [in_dim, out_dim/tp_size] per rank
    - Row split: weight [in_dim, out_dim] -> [in_dim/tp_size, out_dim] per rank
    """

    def __init__(
        self,
        weight_name,
        bias_name,
        mm_type="Default",
        tp_group=None,
        tp_rank=0,
        tp_size=1,
        split_dim="col",
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
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.split_dim = split_dim  # "col" for column split, "row" for row split
        assert split_dim in ["col", "row"], f"split_dim must be 'col' or 'row', got {split_dim}"

        self._mm = MM_WEIGHT_REGISTER.get(mm_type, MMWeight)(
            weight_name=weight_name,
            bias_name=bias_name,
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=create_cpu_buffer,
            lazy_load=lazy_load,
            lazy_load_file=lazy_load_file,
            is_post_adapter=is_post_adapter,
            lora_prefix=lora_prefix,
            lora_path=lora_path,
        )
        self._row_split_bias = None

    def load(self, weight_dict):
        """Load weights using internal MMWeight's load method.

        Note: Weights in weight_dict are already split by _load_weights_from_rank0.
        The format is [out_dim/tp_size, in_dim] for column split or [out_dim, in_dim/tp_size] for row split.
        MMWeight.load will handle the transposition via create_default_tensors.

        For row split, bias is not split and should be added after all-reduce.
        We temporarily remove bias from _mm to prevent it from being added before all-reduce.
        """
        self._mm.load(weight_dict)
        if self.split_dim == "row" and self.bias_name is not None and self.bias_name in weight_dict:
            self._row_split_bias = self._mm.bias.clone()
            self._mm.bias = None

    def apply(self, input_tensor):
        """Apply matrix multiplication with tensor parallel support."""
        # Use internal MMWeight's apply method (handles fp8, int8, etc.)
        # For row split, _mm.bias is None, so bias won't be added here
        output = self._mm.apply(input_tensor)

        # For row split, need all-reduce to combine results from all ranks
        if self.split_dim == "row" and self.tp_size > 1 and self.tp_group is not None:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
            # Add bias after all-reduce (bias is not split for row split)
            if self._row_split_bias is not None:
                output = output + self._row_split_bias

        return output
