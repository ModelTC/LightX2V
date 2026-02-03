from lightx2v_platform.ops.mm.template import MMWeightQuantTemplate
from lightx2v_platform.registry_factory import PLATFORM_MM_WEIGHT_REGISTER

try:
    import torch_npu
except ImportError:
    torch_npu = None


@PLATFORM_MM_WEIGHT_REGISTER("int8-npu")
class MMWeightWint8channelAint8channeldynamicNpu(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-Npu

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: npu
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
