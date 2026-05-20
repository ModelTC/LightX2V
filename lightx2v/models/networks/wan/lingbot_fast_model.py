import torch

from lightx2v.models.networks.wan.infer.lingbot_fast.pre_infer import WanLingbotFastPreInfer
from lightx2v.models.networks.wan.infer.lingbot_fast.transformer_infer import WanLingbotFastTransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.lingbot_model import WanLingbotModel


class WanLingbotFastModel(WanLingbotModel):
    def __init__(self, model_path, config, device, model_type="wan2.1", lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, model_type, lora_path, lora_strength)

    def _init_infer_class(self):
        self.pre_infer_class = WanLingbotFastPreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = WanLingbotFastTransformerInfer

    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        current_start_frame = self.scheduler.seg_index * self.scheduler.num_frame_per_chunk
        current_end_frame = (self.scheduler.seg_index + 1) * self.scheduler.num_frame_per_chunk
        noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)

        self.scheduler.noise_pred[:, current_start_frame:current_end_frame] = noise_pred

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()
