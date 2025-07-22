import os
import torch
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.post_weights import WanPostWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.causvid.transformer_infer import (
    WanTransformerInferCausVid,
)
from lightx2v.utils.envs import *
from safetensors import safe_open


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

    def _load_ckpt(self, use_bf16, skip_bf16):
        ckpt_folder = "causvid_models"
        safetensors_path = os.path.join(self.model_path, f"{ckpt_folder}/causal_model.safetensors")
        if os.path.exists(safetensors_path):
            with safe_open(safetensors_path, framework="pt") as f:
                weight_dict = {key: (f.get_tensor(key).to(torch.bfloat16) if use_bf16 or all(s not in key for s in skip_bf16) else f.get_tensor(key)).pin_memory().to(self.device) for key in f.keys()}
                return weight_dict

        ckpt_path = os.path.join(self.model_path, f"{ckpt_folder}/causal_model.pt")
        if os.path.exists(ckpt_path):
            weight_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            weight_dict = {
                key: (weight_dict[key].to(torch.bfloat16) if use_bf16 or all(s not in key for s in skip_bf16) else weight_dict[key]).pin_memory().to(self.device) for key in weight_dict.keys()
            }
            return weight_dict

        return super()._load_ckpt(use_bf16, skip_bf16)

    @torch.no_grad()
    def infer(self, inputs, kv_start, kv_end):
        if self.config["cpu_offload"]:
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()

        embed, grid_sizes, pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, positive=True, kv_start=kv_start, kv_end=kv_end)

        x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out, kv_start, kv_end)
        self.scheduler.noise_pred = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

        if self.config["cpu_offload"]:
            self.pre_weight.to_cpu()
            self.post_weight.to_cpu()
