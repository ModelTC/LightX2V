from lightx2v_platform.ops.norm.norm_template import RMSWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_RMS_WEIGHT_REGISTER

try:
    import ixformer.inference.functions as ixf
except ImportError:
    ixf = None


@PLATFORM_RMS_WEIGHT_REGISTER("iluvatar_rms_norm")
class IluvatarRmsNormWeight(RMSWeightTemplate):
    def __init__(self, weight_name, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False, eps=0.000001):
        super().__init__(weight_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, is_post_adapter, eps)
        assert ixf is not None, "iluvatar ixformer is not installed."

    def apply(self, input_tensor):
        output = ixf.rms_norm(input_tensor.contiguous(), self.weight, self.eps)
        return output
