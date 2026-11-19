from collections.abc import MutableMapping

import torch

from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.ops.mm.template import MMWeightPerChannelQuantTemplate, MMWeightQuantTemplate
from lightx2v_platform.registry_factory import PLATFORM_MM_WEIGHT_REGISTER

try:
    import torch_npu
except ImportError:
    torch_npu = None


@PLATFORM_MM_WEIGHT_REGISTER("int8-npu")
class MMWeightWint8channelAint8channeldynamicNpu(MMWeightPerChannelQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Npu

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: npu
    """

    checkpoint_dtype = torch.int8

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
        super().__init__(weight_name, bias_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, is_post_adapter, lora_prefix, lora_path)
        self.load_func = self.load_int8_perchannel_sym
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_int8_perchannel_sym_npu

    def act_quant_int8_perchannel_sym_npu(self, x):
        input_tensor_quant, input_tensor_scale = torch_npu.npu_dynamic_quant(x)
        return input_tensor_quant, input_tensor_scale

    def apply(self, input_tensor):
        dtype = input_tensor.dtype
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = torch_npu.npu_quant_matmul(
            input_tensor_quant, self.weight, self.weight_scale.reshape(-1), offset=None, bias=self.bias, pertoken_scale=input_tensor_scale.reshape(-1), output_dtype=dtype
        )
        return output_tensor


@PLATFORM_MM_WEIGHT_REGISTER("mxfp8-npu")
class MMWeightWmxfp8Amxfp8dynamicNpu(MMWeightQuantTemplate):
    """Online W8A8 MXFP8 for Ascend NPU using group-32 E8M0 scales."""

    @staticmethod
    def validate_runtime():
        """Check the required TorchNPU interfaces before loading the checkpoint."""
        if AI_DEVICE != "npu":
            raise RuntimeError("mxfp8-npu requires an Ascend NPU device")
        required = ("npu_dynamic_mx_quant", "npu_quant_matmul", "float8_e8m0fnu")
        if torch_npu is None or not all(hasattr(torch_npu, name) for name in required):
            raise RuntimeError("mxfp8-npu requires TorchNPU with MXFP8 quantization, matmul and E8M0 support")

    def load(self, weight_dict):
        if self.create_cuda_buffer or self.create_cpu_buffer or self.lazy_load or not isinstance(weight_dict, MutableMapping):
            raise ValueError("mxfp8-npu supports online loading from resident BF16/FP16 checkpoints only")
        self.validate_runtime()
        weight = weight_dict[self.weight_name]
        if weight.dim() != 2:
            raise ValueError(f"mxfp8-npu expects a 2D weight, got {tuple(weight.shape)} for {self.weight_name}")
        if weight.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(f"mxfp8-npu expects BF16/FP16 weights, got {weight.dtype} for {self.weight_name}")
        if weight.shape[1] % 32 != 0:
            raise ValueError(f"mxfp8-npu requires the K dimension to be divisible by 32 for {self.weight_name}")

        bias = None
        if self.bias_name is not None:
            bias = weight_dict[self.bias_name].to(AI_DEVICE, dtype=torch.float32)
        weight = weight.to(AI_DEVICE).contiguous()
        weight_fp8, weight_scale = torch_npu.npu_dynamic_mx_quant(weight, dst_type=torch.float8_e4m3fn)
        # TorchNPU returns padded scale pairs with shape [N, ceil(K / 64), 2].
        weight_fp8 = weight_fp8.transpose(0, 1).contiguous()
        weight_scale = weight_scale.transpose(0, 1).contiguous()

        self.weight = weight_fp8
        self.weight_scale = weight_scale
        self.bias = bias
        # Release the checkpoint tensors once this layer's quantized weights are ready.
        del weight_dict[self.weight_name]
        if self.bias_name is not None:
            del weight_dict[self.bias_name]

    def apply(self, input_tensor):
        if input_tensor.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(f"mxfp8-npu expects BF16/FP16 activations, got {input_tensor.dtype}")
        if input_tensor.shape[-1] != self.weight.shape[0]:
            raise ValueError(f"mxfp8-npu input K={input_tensor.shape[-1]} does not match weight K={self.weight.shape[0]}")

        original_shape = input_tensor.shape
        input_tensor = input_tensor.reshape(-1, original_shape[-1])
        input_quant, input_scale = torch_npu.npu_dynamic_mx_quant(input_tensor, dst_type=torch.float8_e4m3fn)
        bias = self._get_actual_bias()
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        scale_dtype = torch_npu.float8_e8m0fnu
        output = torch_npu.npu_quant_matmul(
            input_quant,
            self.weight,
            self.weight_scale,
            scale_dtype=scale_dtype,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=scale_dtype,
            bias=bias,
            output_dtype=input_tensor.dtype,
            group_sizes=[1, 1, 32],
        )
        return output.reshape(*original_shape[:-1], self.weight.shape[1])
