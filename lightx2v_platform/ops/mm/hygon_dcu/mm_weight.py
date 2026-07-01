import os

import torch
import torch.nn.functional as F

from lightx2v_platform.ops.mm.template import MMWeightQuantTemplate
from lightx2v_platform.registry_factory import PLATFORM_MM_WEIGHT_REGISTER

try:
    from vllm import _custom_ops as ops
except ImportError:
    ops = None

try:
    from lightx2v.utils.quant_utils import IntegerQuantizer
except ImportError:
    IntegerQuantizer = None

try:
    from lmslim.quantize.quant_ops import hipblaslt_w8a8_channelwise_gemm
except ImportError:
    hipblaslt_w8a8_channelwise_gemm = None


def _load_auto_quant_bias(module, weight_dict):
    module.bias = None
    module.pin_bias = None
    if module.bias_name is None or module.bias_name not in weight_dict:
        return
    bias = weight_dict[module.bias_name]
    if module.bias_force_fp32:
        bias = bias.to(torch.float32)
    elif hasattr(module, "infer_dtype"):
        bias = bias.to(module.infer_dtype)
    weight = getattr(module, "weight", None)
    target_device = weight.device if weight is not None else bias.device
    module.bias = bias.to(target_device)


def _make_weight_contiguous(module):
    if hasattr(module, "weight") and module.weight is not None:
        module.weight = module.weight.contiguous()
    if hasattr(module, "weight_scale") and module.weight_scale is not None:
        module.weight_scale = module.weight_scale.contiguous()


def _flatten_last_dim(input_tensor):
    if input_tensor.dim() == 2:
        return input_tensor, None
    original_shape = input_tensor.shape
    return input_tensor.reshape(-1, original_shape[-1]), original_shape[:-1]


def _restore_last_dim(output_tensor, prefix_shape):
    if prefix_shape is None:
        return output_tensor
    return output_tensor.reshape(*prefix_shape, output_tensor.shape[-1])


def _bias_or_none(module, out_dtype=None):
    if hasattr(module, "bias") and module.bias is not None:
        bias = module.bias
        if out_dtype is not None and bias.dtype != out_dtype:
            bias = bias.to(out_dtype)
        return bias
    return None


def _env_flag(name, default="0"):
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_patterns(name, default=""):
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.replace(";", ",").split(",") if item.strip())


def _matches_any(name, patterns):
    return any(pattern in name for pattern in patterns)


def _use_selective_bf16_fallback(weight_name):
    if not _env_flag("LIGHTX2V_INT8_SELECTIVE"):
        return False
    include = _env_patterns("LIGHTX2V_INT8_SELECTIVE_INCLUDE")
    exclude = _env_patterns(
        "LIGHTX2V_INT8_SELECTIVE_EXCLUDE",
        "txt_branch",
    )
    if include and not _matches_any(weight_name, include):
        return True
    return _matches_any(weight_name, exclude)


def _require_hipblaslt_w8a8_channelwise_gemm():
    if hipblaslt_w8a8_channelwise_gemm is None:
        raise RuntimeError("int8-vllm-hygon-dcu requires lmslim.quantize.quant_ops.hipblaslt_w8a8_channelwise_gemm on Hygon DCU.")
    return hipblaslt_w8a8_channelwise_gemm


@PLATFORM_MM_WEIGHT_REGISTER("int8-vllm-hygon-dcu")
class MMWeightWint8channelAint8channeldynamicVllmHygonDcu(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Vllm

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: vllm (blaslt for ROCm/DCU)
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
        super().__init__(weight_name, bias_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, is_post_adapter)
        self.load_func = self.load_int8_perchannel_sym
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_int8_perchannel_sym_vllm
        self.use_bf16_fallback = _use_selective_bf16_fallback(weight_name)

    def load(self, weight_dict):
        super().load(weight_dict)
        _make_weight_contiguous(self)

    def load_int8_perchannel_sym(self, weight_dict):
        """Load INT8 per-channel symmetric quantized weights."""
        if self.use_bf16_fallback:
            if not self.config.get("weight_auto_quant", False):
                raise RuntimeError("Selective BF16 fallback requires weight_auto_quant=1 so original BF16 weights are available.")
            self.weight = weight_dict[self.weight_name].to(self.infer_dtype)
            self.weight_scale = None
            _load_auto_quant_bias(self, weight_dict)
            return

        if self.config.get("weight_auto_quant", False):
            if IntegerQuantizer is None:
                raise ImportError("IntegerQuantizer not available. Please ensure lightx2v.utils.quant_utils is available.")
            self.weight = weight_dict[self.weight_name].to(torch.float32)
            w_quantizer = IntegerQuantizer(8, True, "per_channel")
            self.weight, self.weight_scale, _ = w_quantizer.real_quant_tensor(self.weight)
            self.weight = self.weight.to(torch.int8)
            self.weight_scale = self.weight_scale.to(torch.float32)
            _load_auto_quant_bias(self, weight_dict)
        else:
            self.load_quantized(weight_dict)

    def act_quant_int8_perchannel_sym_vllm(self, x):
        """Activation quantization using vLLM's scaled_int8_quant."""
        if ops is None:
            raise ImportError("vLLM _custom_ops not available. Please install vLLM.")
        input_tensor_quant, input_tensor_scale, _ = ops.scaled_int8_quant(x, scale=None, azp=None, symmetric=True)
        return input_tensor_quant, input_tensor_scale

    def prepare_quantized_input(self, input_tensor):
        if self.use_bf16_fallback:
            raise RuntimeError("BF16 fallback weights do not support shared INT8 activation quantization.")
        input_2d, prefix_shape = _flatten_last_dim(input_tensor)
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_2d)
        return input_2d, prefix_shape, input_tensor_quant, input_tensor_scale

    def _apply_quantized_2d(self, input_2d, input_tensor_quant, input_tensor_scale, out_dtype):
        m, k = input_tensor_quant.shape
        n = self.weight.shape[0]
        hipblaslt_gemm = _require_hipblaslt_w8a8_channelwise_gemm()
        _, output_tensor = hipblaslt_gemm(
            a=input_tensor_quant.contiguous(),
            b=self.weight,
            scale_a=input_tensor_scale.contiguous(),
            scale_b=self.weight_scale,
            m=m,
            n=n,
            k=k,
            transpose_flag="NT",
            out_dtype=out_dtype,
            bias=_bias_or_none(self, out_dtype),
        )
        output_tensor = output_tensor.reshape(-1, n).narrow(0, 0, input_2d.shape[0])
        return output_tensor

    def apply_quantized_input(self, input_tensor, quantized_input):
        if self.use_bf16_fallback:
            return self.apply(input_tensor)
        dtype = input_tensor.dtype
        out_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        input_2d, prefix_shape, input_tensor_quant, input_tensor_scale = quantized_input
        output_tensor = self._apply_quantized_2d(input_2d, input_tensor_quant, input_tensor_scale, out_dtype)

        if output_tensor.dtype != dtype:
            output_tensor = output_tensor.to(dtype)
        return _restore_last_dim(output_tensor, prefix_shape)

    def _apply_bf16(self, input_tensor):
        if self.weight.dtype != input_tensor.dtype:
            self.weight = self.weight.to(input_tensor.dtype)
        bias = _bias_or_none(self, input_tensor.dtype)
        return F.linear(input_tensor, self.weight, bias)

    def apply(self, input_tensor):
        if self.use_bf16_fallback:
            return self._apply_bf16(input_tensor)
        return self.apply_quantized_input(input_tensor, self.prepare_quantized_input(input_tensor))
