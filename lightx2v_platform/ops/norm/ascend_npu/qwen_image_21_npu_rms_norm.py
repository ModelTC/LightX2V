import torch

from lightx2v_platform.ops.norm.norm_template import RMSWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_RMS_WEIGHT_REGISTER

try:
    import torch_npu
except ImportError:
    torch_npu = None


def _is_npu_tensor(tensor):
    return tensor.device.type == "npu"


@PLATFORM_RMS_WEIGHT_REGISTER("qwen_image_21_npu_rms_norm")
class QwenImage21NpuRmsNormWeight(RMSWeightTemplate):
    """Qwen-Image-2.1 Q/K RMSNorm backed by ``npu_rms_norm``.

    ``fp32_variance`` rounds the normalized activation to the BF16 weight dtype
    before applying the learned scale.  Passing that scale directly to the NPU
    operator would remove this rounding boundary.  Use a cached unit gamma in
    the native operator and multiply the learned scale afterwards to preserve
    the model's arithmetic contract.  The fallback mirrors ``fp32_variance``
    when torch-npu is unavailable.
    """

    def _get_actual_weight(self):
        weight = self.weight
        weight_diff = getattr(self, "weight_diff", None)
        if weight is None or weight_diff is None:
            return weight
        if weight_diff.device != weight.device or weight_diff.dtype != weight.dtype:
            weight_diff = weight_diff.to(device=weight.device, dtype=weight.dtype)
        return weight + weight_diff

    def _fallback(self, input_tensor, weight):
        input_dtype = input_tensor.dtype
        variance = input_tensor.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = input_tensor * torch.rsqrt(variance + self.eps)
        if weight is not None:
            if weight.dtype in (torch.float16, torch.bfloat16):
                hidden_states = hidden_states.to(weight.dtype)
            hidden_states = hidden_states * weight
        return hidden_states.to(input_dtype)

    def apply(self, input_tensor):
        weight = self._get_actual_weight()
        if (
            torch_npu is None
            or not hasattr(torch_npu, "npu_rms_norm")
            or weight is None
            or not _is_npu_tensor(input_tensor)
            or input_tensor.dtype != weight.dtype
        ):
            return self._fallback(input_tensor, weight)

        unit_weight = getattr(self, "_unit_weight", None)
        if unit_weight is None or unit_weight.shape != weight.shape or unit_weight.device != weight.device or unit_weight.dtype != weight.dtype:
            unit_weight = torch.ones_like(weight)
            self._unit_weight = unit_weight
        output, _ = torch_npu.npu_rms_norm(input_tensor, unit_weight, self.eps)
        return (output * weight).to(input_tensor.dtype)
