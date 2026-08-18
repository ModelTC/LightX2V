import torch

from lightx2v_platform.ops.norm.norm_template import RMSWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_RMS_WEIGHT_REGISTER

try:
    import sycl_kernels as _sycl_kernels

    _rms_norm = _sycl_kernels.rms_norm
    _has_rms_norm = _sycl_kernels.has_rms_norm()
except (ImportError, RuntimeError):
    _has_rms_norm = False
    _rms_norm = None


@PLATFORM_RMS_WEIGHT_REGISTER("intel_xpu")
class IntelXpuRMSWeight(RMSWeightTemplate):
    """ESIMD RMSNorm with a torch fallback for unsupported contracts."""

    def apply(self, input_tensor):
        weight = getattr(self, "weight", None)
        if weight is None:
            return torch.nn.functional.rms_norm(input_tensor, (input_tensor.shape[-1],), eps=self.eps)

        hidden_size = input_tensor.shape[-1]
        use_esimd = (
            _has_rms_norm
            and _rms_norm is not None
            and input_tensor.device.type == "xpu"
            and weight.device == input_tensor.device
            and input_tensor.dtype == weight.dtype
            and input_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and 0 < hidden_size <= 8192
            and hidden_size % 32 == 0
            and self.sensitive_layer_dtype == self.infer_dtype
        )
        if use_esimd:
            original_shape = input_tensor.shape
            flat_input = input_tensor.contiguous().view(-1, hidden_size)
            return _rms_norm(weight.contiguous(), flat_input, self.eps).view(original_shape)

        compute_input = input_tensor
        compute_weight = weight
        if self.sensitive_layer_dtype != self.infer_dtype:
            compute_input = input_tensor.float()
            compute_weight = weight.float()
        return torch.nn.functional.rms_norm(
            compute_input,
            (hidden_size,),
            weight=compute_weight,
            eps=self.eps,
        ).to(input_tensor.dtype)
