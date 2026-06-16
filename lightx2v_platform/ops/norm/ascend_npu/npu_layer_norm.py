import torch

from lightx2v_platform.ops.norm.norm_template import LayerNormWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_LAYERNORM_WEIGHT_REGISTER

try:
    import torch_npu
except ImportError:
    torch_npu = None


@PLATFORM_LAYERNORM_WEIGHT_REGISTER("npu_layer_norm")
class NpuLayerNormWeight(LayerNormWeightTemplate):
    def __init__(self, weight_name=None, bias_name=None, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False, eps=1e-6, **kwargs):
        super().__init__(weight_name, bias_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, is_post_adapter, eps)

    def apply(self, input_tensor):
        if torch_npu is not None and hasattr(torch_npu, 'npu_layer_norm'):
            out = torch_npu.npu_layer_norm(
                input_tensor,
                (input_tensor.shape[-1],),
                self.weight,
                self.bias,
                self.eps,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out
        output = torch.nn.functional.layer_norm(
            input_tensor,
            (input_tensor.shape[-1],),
            self.weight,
            self.bias,
            self.eps,
        )
        return output