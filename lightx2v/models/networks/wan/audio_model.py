import glob
import os
import time

import torch
from safetensors import safe_open

from lightx2v.common.ops.attn.radial_attn import MaskMap
from lightx2v.models.networks.wan.infer.audio.post_wan_audio_infer import WanAudioPostInfer
from lightx2v.models.networks.wan.infer.audio.pre_wan_audio_infer import WanAudioPreInfer
from lightx2v.models.networks.wan.infer.feature_caching.transformer_infer import WanTransformerInferTeaCaching
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.post_weights import WanPostWeights
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)


class WanAudioModel(WanModel):
    pre_weight_class = WanPreWeights
    post_weight_class = WanPostWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = WanAudioPreInfer
        self.post_infer_class = WanAudioPostInfer

    @torch.no_grad()
    def infer(self, inputs):
        if self.config["cpu_offload"]:
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()

        if self.transformer_infer.mask_map is None:
            _, c, h, w = self.scheduler.latents.shape
            num_frame = c + 1  # for r2v
            video_token_num = num_frame * (h // 2) * (w // 2)
            self.transformer_infer.mask_map = MaskMap(video_token_num, num_frame)

        embed, grid_sizes, pre_infer_out, valid_patch_length = self.pre_infer.infer(self.pre_weight, inputs, positive=True)
        x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out)
        noise_pred_cond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes, valid_patch_length)[0]

        if self.config["feature_caching"] == "Tea":
            self.scheduler.cnt += 1
            if self.scheduler.cnt >= self.scheduler.num_steps:
                self.scheduler.cnt = 0
        self.scheduler.noise_pred = noise_pred_cond

        if self.config["enable_cfg"]:
            embed, grid_sizes, pre_infer_out, valid_patch_length = self.pre_infer.infer(self.pre_weight, inputs, positive=False)
            x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out)
            noise_pred_uncond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes, valid_patch_length)[0]

            if self.config["feature_caching"] == "Tea":
                self.scheduler.cnt += 1
                if self.scheduler.cnt >= self.scheduler.num_steps:
                    self.scheduler.cnt = 0

            self.scheduler.noise_pred = noise_pred_uncond + self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

            if self.config["cpu_offload"]:
                self.pre_weight.to_cpu()
                self.post_weight.to_cpu()


class Wan22MoeAudioModel(WanAudioModel):
    def _load_ckpt(self, use_bf16, skip_bf16):
        safetensors_files = glob.glob(os.path.join(self.model_path, "*.safetensors"))
        weight_dict = {}
        for file_path in safetensors_files:
            file_weights = self._load_safetensor_to_dict(file_path, use_bf16, skip_bf16)
            weight_dict.update(file_weights)
        return weight_dict
