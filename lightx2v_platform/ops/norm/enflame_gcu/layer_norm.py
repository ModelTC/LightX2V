import torch

from lightx2v_platform.ops.norm.norm_template import LayerNormWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_LAYERNORM_WEIGHT_REGISTER


@PLATFORM_LAYERNORM_WEIGHT_REGISTER("gcu_layer_norm")
class GcuLayerNormWeight(LayerNormWeightTemplate):
    def __init__(self, weight_name=None, bias_name=None, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False, eps=1e-6):
        super().__init__(weight_name, bias_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, is_post_adapter, eps)

    def apply(self, input_tensor):
        # GCU does not support mixed precision (Float input with Half weight/bias)
        # Explicitly convert weight and bias to float32 to avoid dtype mismatch in autocast context
        x_float = input_tensor.float()
        # Convert weight and bias to float32 if they exist
        if self.weight is not None:
            weight_float = self.weight.float()
        else:
            weight_float = None
        if self.bias is not None:
            bias_float = self.bias.float()
        else:
            bias_float = None
        # Use functional layer_norm with explicit float32 parameters
        result = torch.nn.functional.layer_norm(x_float, (input_tensor.shape[-1],), weight_float, bias_float, self.eps)
        return result.type_as(input_tensor)
