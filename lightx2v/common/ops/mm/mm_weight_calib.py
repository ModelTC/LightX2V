import torch

from lightx2v.utils.quant_utils import FloatQuantizer, IntegerQuantizer
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER

from .mm_weight import MMWeight


@MM_WEIGHT_REGISTER("Calib")
class MMWeightCalib(MMWeight):
    def __init__(self, weight_name, bias_name):
        super().__init__(weight_name, bias_name)

    def load(self, weight_dict):
        assert self.config and self.config.get("mm_type", "Default") != "Default"
        self.weight = weight_dict[self.weight_name]
        self.get_quantizer()
        shape_and_dtype = self.get_quant_shape_and_dtype(self.weight.shape)
        self.realq_weight, self.scales, self.zeros = self.w_quantizer.real_quant_tensor(self.weight)
        self.realq_weight = self.realq_weight.view(shape_and_dtype["tensor"][0]).contiguous().to(shape_and_dtype["tensor"][1])
        self.scales = self.scales.view(shape_and_dtype["scales"][0]).contiguous().to(shape_and_dtype["scales"][1])
        if self.zeros is not None:
            self.zeros = self.zeros.view(shape_and_dtype["zeros"][0]).contiguous().to(shape_and_dtype["zeros"][1])

    def apply(self, input_tensor):
        return super().apply(input_tensor)

    def get_quantizer(self):
        if self.config["mm_type"] == "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm":
            self.w_setting = {"bit": "e4m3", "symmetric": True, "granularity": "per_channel"}
            self.a_setting = {"bit": "e4m3", "symmetric": True, "granularity": "per_channel"}
            self.w_quantizer = FloatQuantizer(**self.w_setting)
            self.a_quantizer = FloatQuantizer(**self.a_setting)
            self.act_dynamic_quant = True
        else:
            raise NotImplementedError(f"Unsupported mm_type: {self.config['mm_type']}")

    def get_quant_shape_and_dtype(self, shape):
        if self.config["mm_type"] == "W-fp8-channel-sym-A-fp8-channel-sym-dynamic-Vllm":
            return {
                "tensor": (shape, torch.float8_e5m2),
                "scales": ((shape[0], 1), torch.float32),
                "zeros": None,
            }
        else:
            raise NotImplementedError(f"Unsupported mm_type: {self.config['mm_type']}")
