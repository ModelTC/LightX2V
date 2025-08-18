import os

import torch

from lightx2v.common.ops.attn.radial_attn import MaskMap
from lightx2v.models.networks.wan.infer.causvid.transformer_infer import (
    WanTransformerInferCausVid,
)
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.post_weights import WanPostWeights
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.utils.envs import *
from lightx2v.utils.utils import find_torch_model_path


class WanCausVidModel(WanModel):
    pre_weight_class = WanPreWeights
    post_weight_class = WanPostWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)

    def _init_infer_class(self):
        self.pre_infer_class = WanPreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = WanTransformerInferCausVid

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        ckpt_path = find_torch_model_path(self.config, self.model_path, "causvid_model.pt")
        if os.path.exists(ckpt_path):
            weight_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            weight_dict = {
                key: (weight_dict[key].to(GET_DTYPE()) if unified_dtype or all(s not in key for s in sensitive_layer) else weight_dict[key].to(GET_SENSITIVE_DTYPE())).pin_memory().to(self.device)
                for key in weight_dict.keys()
            }
            return weight_dict

        return super()._load_ckpt(unified_dtype, sensitive_layer)

    @torch.no_grad()
    def infer(self, inputs, kv_start, kv_end):
        if self.transformer_infer.mask_map is None:
            _, c, h, w = self.scheduler.latents.shape
            video_token_num = c * (h // 2) * (w // 2)
            self.transformer_infer.mask_map = MaskMap(video_token_num, c)

        if self.config["cpu_offload"]:
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()

        embed, grid_sizes, pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, kv_start=kv_start, kv_end=kv_end)

        x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out, kv_start, kv_end)
        self.scheduler.noise_pred = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

        if self.config["cpu_offload"]:
            self.pre_weight.to_cpu()
            self.post_weight.to_cpu()
