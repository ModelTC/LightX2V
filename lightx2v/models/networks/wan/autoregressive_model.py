import torch

from lightx2v.models.networks.autoregressive_model import AutoregressiveModelMixin
from lightx2v.models.networks.wan.model import WanModel


class WanAutoregressiveModel(AutoregressiveModelMixin, WanModel):
    """Run a chunk and place its prediction in the scheduler's full sequence."""

    def _apply_weights(self, weight_dict=None):
        super()._apply_weights(weight_dict)
        self.prepare_autoregressive_projection_groups()

    def clear_autoregressive_cache(self):
        infer = getattr(self, "transformer_infer", None)
        if infer is not None and infer.cuda_graph_stack is not None:
            infer.cuda_graph_stack.clear()
        super().clear_autoregressive_cache()

    def _infer_cond_uncond(self, inputs, infer_condition=True):
        infer = self.transformer_infer
        text = inputs["text_encoder_output"]["context" if infer_condition else "context_null"]
        infer.set_cross_attn_context(text, branch=infer_condition)
        if self.config.get("use_image_encoder", True) and self.config["task"] in ("i2v", "flf2v", "animate", "s2v", "rs2v"):
            infer.set_cross_attn_context(inputs["image_encoder_output"]["clip_encoder_out"], branch=infer_condition, kind="image")
        return super()._infer_cond_uncond(inputs, infer_condition)

    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            if self.offload_granularity == "model" and (self.scheduler.step_index == 0 or self.scheduler.is_rerun):
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        current_start_frame = self.scheduler.seg_index * self.scheduler.num_frame_per_chunk
        current_end_frame = (self.scheduler.seg_index + 1) * self.scheduler.num_frame_per_chunk
        noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)

        if noise_pred is not None:
            self.scheduler.noise_pred[:, current_start_frame:current_end_frame] = noise_pred
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()
