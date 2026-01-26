import torch

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

    def load_int8_perchannel_sym(self, weight_dict):
        """Load INT8 per-channel symmetric quantized weights."""
        if self.config.get("weight_auto_quant", False):
            if IntegerQuantizer is None:
                raise ImportError("IntegerQuantizer not available. Please ensure lightx2v.utils.quant_utils is available.")
            self.weight = weight_dict[self.weight_name].to(torch.float32)
            w_quantizer = IntegerQuantizer(8, True, "per_channel")
            self.weight, self.weight_scale, _ = w_quantizer.real_quant_tensor(self.weight)
            self.weight = self.weight.to(torch.int8)
            self.weight_scale = self.weight_scale.to(torch.float32)
        else:
            self.load_quantized(weight_dict)

    def act_quant_int8_perchannel_sym_vllm(self, x):
        """Activation quantization using vLLM's scaled_int8_quant."""
        if ops is None:
            raise ImportError("vLLM _custom_ops not available. Please install vLLM.")
        input_tensor_quant, input_tensor_scale, _ = ops.scaled_int8_quant(x, scale=None, azp=None, symmetric=True)
        return input_tensor_quant, input_tensor_scale

    def apply(self, input_tensor):
        shape = (input_tensor.shape[0], self.weight.shape[1])
        dtype = input_tensor.dtype
        device = input_tensor.device

        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)

        # Use ops.blaslt_scaled_mm from vllm for ROCm/DCU instead of torch.ops._C.cutlass_scaled_mm
        if ops is not None and hasattr(ops, "blaslt_scaled_mm"):
            # Ensure out_dtype is bfloat16 or float16 as required by blaslt_scaled_mm
            out_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16

            # Ensure input tensor is contiguous for optimal performance
            input_tensor_quant = input_tensor_quant.contiguous()

            output_tensor = ops.blaslt_scaled_mm(
                input_tensor_quant,
                self.weight,
                input_tensor_scale,
                self.weight_scale,
                out_dtype,
                self.bias if self.bias is not None else None,
            )

            # Convert back to original dtype if needed
            if output_tensor.dtype != dtype:
                output_tensor = output_tensor.to(dtype)
        else:
            # Fallback: use manual dequantization and matmul
            input_dequant = input_tensor_quant.to(dtype) * input_tensor_scale.to(dtype)
            weight_dequant = self.weight.to(dtype) * self.weight_scale.to(dtype)
            output_tensor = torch.matmul(input_dequant, weight_dequant)
            if self.bias is not None:
                output_tensor = output_tensor + self.bias

        return output_tensor
