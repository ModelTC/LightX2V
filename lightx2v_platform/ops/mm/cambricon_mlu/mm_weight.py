import torch

from lightx2v.utils.quant_utils import IntegerQuantizer
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

    def load_quantized(self, weight_dict):
        if self.create_cuda_buffer:
            # move to cuda buffer
            self.weight_cuda_buffer = weight_dict[self.weight_name].cuda()
            self.weight_scale_cuda_buffer = weight_dict[self.weight_scale_name].float().cuda()
        else:
            device = weight_dict[self.weight_name].device
            if device.type == "cpu":
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
                self.weight = weight_dict[self.weight_name]
                self.weight_scale = weight_dict[self.weight_scale_name].float()

        if self.bias_name is not None:
            if self.create_cuda_buffer:
                # move to cuda buffer
                self.bias_cuda_buffer = weight_dict[self.bias_name].cuda()
            else:
                device = weight_dict[self.bias_name].device
                if device.type == "cpu":
                    bias_shape = weight_dict[self.bias_name].shape
                    bias_dtype = weight_dict[self.bias_name].dtype
                    self.pin_bias = torch.empty(bias_shape, pin_memory=True, dtype=bias_dtype)
                    self.pin_bias.copy_(weight_dict[self.bias_name])
                else:
                    self.bias = weight_dict[self.bias_name]
        else:
            self.bias = None
            self.pin_bias = None

    def load_int8_perchannel_sym(self, weight_dict):
        if self.config.get("weight_auto_quant", False):
            self.weight = weight_dict[self.weight_name].to(torch.float32)
            w_quantizer = IntegerQuantizer(8, True, "per_channel")
            self.weight, self.weight_scale, _ = w_quantizer.real_quant_tensor(self.weight)
            self.weight = self.weight.to(torch.int8)
            self.weight_scale = self.weight_scale.to(torch.float32)
        else:
            self.load_quantized(weight_dict)

    def act_quant_int8_perchannel_sym_tmo(self, x):
        input_tensor_quant, input_tensor_scale = tmo.scaled_quantize(x)
        return input_tensor_quant, input_tensor_scale

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

    def apply(self, input_tensor):
        dtype = input_tensor.dtype
        input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
        output_tensor = tmo.scaled_matmul(
            input_tensor_quant, self.weight.contiguous(), input_tensor_scale, self.weight_scale.squeeze(-1), bias=self.bias if self.bias is not None else None, output_dtype=dtype, use_hp_active=True
        )
        return output_tensor
