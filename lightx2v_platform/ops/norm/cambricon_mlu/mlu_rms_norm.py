from lightx2v_platform.ops.norm.norm_template import RMSWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_RMS_WEIGHT_REGISTER

try:
    import torch_mlu_ops as tmo
except ImportError:
    tmo = None


@PLATFORM_RMS_WEIGHT_REGISTER("mlu_rms_norm")
class MluRmsNormWeight(RMSWeightTemplate):
    def __init__(self, weight_name, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False, eps=0.000001):
        super().__init__(weight_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, is_post_adapter, eps)
        assert tmo is not None, "torch_mlu_ops is not installed."

    def apply(self, input_tensor):
        if self.sensitive_layer_dtype != self.infer_dtype:
            output = tmo.fused_rms_norm(
                input_tensor.float(),
                residual=None,
                gamma=self.weight,
                beta=None,
                bias=None,
                eps=self.eps,
                store_output_before_norm=False,
                quant_scale=None,
                out=None,
            ).to(self.infer_dtype)
        else:
            output = tmo.fused_rms_norm(
                input_tensor,
                residual=None,
                gamma=self.weight,
                beta=None,
                bias=None,
                eps=self.eps,
                store_output_before_norm=False,
                quant_scale=None,
                out=None,
            )
        return output
