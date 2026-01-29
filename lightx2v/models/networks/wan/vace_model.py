import torch

from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.vace.transformer_infer import WanVaceTransformerInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.vace.transformer_weights import (
    WanVaceTransformerWeights,
)
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *


class WanVaceModel(WanModel):
    pre_weight_class = WanPreWeights
    transformer_weight_class = WanVaceTransformerWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)

    def _init_infer(self):
        super()._init_infer()
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    def _init_offload_manager(self):
        self.transformer_infer.offload_block_cuda_buffers = self.transformer_weights.offload_block_cuda_buffers
        self.transformer_infer.offload_phase_cuda_buffers = self.transformer_weights.offload_phase_cuda_buffers
        self.transformer_infer.vace_offload_block_cuda_buffers = self.transformer_weights.vace_offload_block_cuda_buffers
        self.transformer_infer.vace_offload_phase_cuda_buffers = self.transformer_weights.vace_offload_phase_cuda_buffers
        if self.lazy_load:
            self.transformer_infer.offload_block_cpu_buffers = self.transformer_weights.offload_block_cpu_buffers
            self.transformer_infer.offload_phase_cpu_buffers = self.transformer_weights.offload_phase_cpu_buffers
            self.transformer_infer.vace_offload_block_cpu_buffers = self.transformer_weights.vace_offload_block_cpu_buffers
            self.transformer_infer.vace_offload_phase_cpu_buffers = self.transformer_weights.vace_offload_phase_cpu_buffers

    def _init_infer_class(self):
        self.pre_infer_class = WanPreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = WanVaceTransformerInfer

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        self.scheduler.infer_condition = infer_condition

        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)
        pre_infer_out.vace_context = inputs["image_encoder_output"]["vae_encoder_out"][0]

        x = self.transformer_infer.infer(self.transformer_weights, pre_infer_out)

        noise_pred = self.post_infer.infer(x, pre_infer_out)[0]

        if self.clean_cuda_cache:
            del x, pre_infer_out
            torch.cuda.empty_cache()

        return noise_pred
