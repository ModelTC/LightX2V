import os
import sys
import torch
import glob
import json
from safetensors import safe_open
from lightx2v.models.networks.wan.model import WanModel, Wan22MoeModel
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.post_weights import WanPostWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.utils.envs import *
from loguru import logger


class WanDistillModel(WanModel):
    pre_weight_class = WanPreWeights
    post_weight_class = WanPostWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)

    def _load_ckpt(self, use_bf16, skip_bf16):
        if self.config.get("enable_dynamic_cfg", False):
            ckpt_path = os.path.join(self.model_path, "distill_cfg_models", "distill_model.safetensors")
        else:
            ckpt_path = os.path.join(self.model_path, "distill_models", "distill_model.safetensors")
        if os.path.exists(ckpt_path):
            logger.info(f"Loading weights from {ckpt_path}")
            return self._load_safetensor_to_dict(ckpt_path, use_bf16, skip_bf16)

        return super()._load_ckpt(use_bf16, skip_bf16)


class Wan22MoeDistillModel(WanDistillModel, Wan22MoeModel):
    def __init__(self, model_path, config, device):
        WanDistillModel.__init__(self, model_path, config, device)

    def _load_ckpt(self, use_bf16, skip_bf16):
        ckpt_path = os.path.join(self.model_path, "distill_model.safetensors")
        if os.path.exists(ckpt_path):
            logger.info(f"Loading weights from {ckpt_path}")
            return self._load_safetensor_to_dict(ckpt_path, use_bf16, skip_bf16)

    @torch.no_grad()
    def infer(self, inputs):
        return Wan22MoeModel.infer(self, inputs)
