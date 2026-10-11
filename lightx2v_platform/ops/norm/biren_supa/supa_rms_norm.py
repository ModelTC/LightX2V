import torch
import torch_br

from lightx2v_platform.ops.norm.norm_template import RMSWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_RMS_WEIGHT_REGISTER


@PLATFORM_RMS_WEIGHT_REGISTER("supa_rms_norm")
class SupaRmsNormWeight(RMSWeightTemplate):
    def __init__(
        self,
        weight_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            eps,
            lora_prefix,
            lora_path,
        )

    def apply(self, input_tensor):
        w = self.weight
        if w.dtype != torch.float32:
            w = w.float()
        return torch_br.supa_rmsnorm_infer(input_tensor.contiguous(), w, self.eps)

    def to_cuda(self, non_blocking=False):
        super().to_cuda(non_blocking=non_blocking)
        if self.weight is not None and self.weight.dtype != torch.float32:
            self.weight = self.weight.float()
