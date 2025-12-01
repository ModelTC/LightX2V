from lightx2v_platform.ops.mm.template import MMWeightQuantTemplate
from lightx2v_platform.registry_factory import PLATFORM_MM_WEIGHT_REGISTER

try:
    import torch_mlu_ops as tmo
except ImportError:
    tmo = None


@PLATFORM_MM_WEIGHT_REGISTER("int8-tmo")
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
        output_tensor = tmo.scaled_matmul(
            input_tensor_quant, self.weight.contiguous(), input_tensor_scale, self.weight_scale.squeeze(-1), bias=self.bias if self.bias is not None else None, output_dtype=dtype, use_hp_active=True
        )
        return output_tensor
