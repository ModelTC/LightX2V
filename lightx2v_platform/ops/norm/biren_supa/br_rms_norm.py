import torch

from lightx2v_platform.ops.norm.norm_template import RMSWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_RMS_WEIGHT_REGISTER

try:
    import torch_br
except ImportError:
    torch_br = None


@PLATFORM_RMS_WEIGHT_REGISTER("biren_rms_norm")
class BirenRmsNormWeight(RMSWeightTemplate):
    def __init__(
        self,
        weight_name="diffusion_model.blocks.0.norm1.weight",
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        if torch_br is None:
            raise ImportError("torch_br is not installed.")
        super().__init__(
            weight_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            eps,
        )
        self._weight_source = None
        self._weight_fp32 = None

    def _get_fp32_weight(self, weight):
        """Convert a norm weight once per device residency interval."""
        if (
            self._weight_source is weight
            and self._weight_fp32 is not None
            and self._weight_fp32.device == weight.device
        ):
            return self._weight_fp32
        if weight.dtype == torch.float32 and weight.is_contiguous():
            converted = weight
        else:
            converted = weight.float().contiguous()
        self._weight_source = weight
        self._weight_fp32 = converted
        return converted

    def to_cuda(self, non_blocking=False):
        super().to_cuda(non_blocking=non_blocking)
        self._weight_source = None
        self._weight_fp32 = None

    def to_cpu(self, non_blocking=False):
        self._weight_source = None
        self._weight_fp32 = None
        super().to_cpu(non_blocking=non_blocking)

    def apply(self, input_tensor):
        weight = self.weight if hasattr(self, "weight") and self.weight is not None else self.weight_cuda_buffer
        w = self._get_fp32_weight(weight)
        x = input_tensor.to(torch.bfloat16) if input_tensor.dtype != torch.bfloat16 else input_tensor
        # supa_rmsnorm requires 3D input [batch, seq, hidden]
        need_squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            need_squeeze = True
        res = torch_br.supa_rmsnorm(x, w, self.eps, training=False)
        if need_squeeze:
            res = res.squeeze(0)
        return res
