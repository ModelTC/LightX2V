import torch

from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightTemplate
from lightx2v.models.networks.minimax_h3.infer.sglang_fused import apply_qk_rms_norm_sglang
from lightx2v.utils.registry_factory import RMS_WEIGHT_REGISTER


@RMS_WEIGHT_REGISTER("h3ref_sgl_qk_rms_norm")
class MiniMaxH3SGLQKRMSNorm(RMSWeightTemplate):
    def apply(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return apply_qk_rms_norm_sglang(input_tensor, self._get_actual_weight(), self.eps)


__all__ = ["MiniMaxH3SGLQKRMSNorm"]
