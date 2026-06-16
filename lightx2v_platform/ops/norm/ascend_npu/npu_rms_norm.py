import torch

from lightx2v_platform.ops.norm.norm_template import RMSWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_RMS_WEIGHT_REGISTER

try:
    import torch_npu
except ImportError:
    torch_npu = None


@PLATFORM_RMS_WEIGHT_REGISTER("npu_rms_norm")
class NpuRmsNormWeight(RMSWeightTemplate):
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def apply(self, input_tensor):
        weight = self.weight
        if torch_npu is not None and hasattr(torch_npu, "npu_rms_norm") and weight is not None:
            if self.sensitive_layer_dtype != self.infer_dtype:
                output_tensor, _ = torch_npu.npu_rms_norm(
                    input_tensor.to(self.sensitive_layer_dtype),
                    weight.to(self.sensitive_layer_dtype),
                    self.eps,
                )
                return output_tensor.to(self.infer_dtype)
            output_tensor, _ = torch_npu.npu_rms_norm(input_tensor, weight, self.eps)
            return output_tensor
        x = self._norm(input_tensor.float())
        if weight is not None:
            return (x.float() * weight.float()).to(self.infer_dtype)
        return x.to(self.infer_dtype)
