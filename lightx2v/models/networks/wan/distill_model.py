import os

import torch
from loguru import logger

from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.post_weights import WanPostWeights
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *


class WanDistillModel(WanModel):
    pre_weight_class = WanPreWeights
    post_weight_class = WanPostWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        # For the old t2v distill model: https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill
        ckpt_path = os.path.join(self.model_path, "distill_model.pt")
        if os.path.exists(ckpt_path):
            logger.info(f"Loading weights from {ckpt_path}")
            weight_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            weight_dict = {
                key: (weight_dict[key].to(GET_DTYPE()) if unified_dtype or all(s not in key for s in sensitive_layer) else weight_dict[key].to(GET_SENSITIVE_DTYPE())).pin_memory().to(self.device)
                for key in weight_dict.keys()
            }
            return weight_dict
        return super()._load_ckpt(unified_dtype, sensitive_layer)
