import json
import os

import torch
from safetensors import safe_open

from lightx2v.models.networks.wan.infer.matrix_game2.pre_infer import WanMtxg2PreInfer
from lightx2v.models.networks.wan.infer.matrix_game2.transformer_infer import WanMtxg2TransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.sf_model import WanSFModel
from lightx2v.models.networks.wan.weights.matrix_game2.pre_weights import WanMtxg2PreWeights
from lightx2v.models.networks.wan.weights.matrix_game2.transformer_weights import WanActionTransformerWeights
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *


class WanSFMtxg2Model(WanSFModel):
    pre_weight_class = WanMtxg2PreWeights
    transformer_weight_class = WanActionTransformerWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        with safe_open(file_path, framework="pt", device=str(self.device)) as f:
            return {key: (f.get_tensor(key).to(GET_DTYPE()) if unified_dtype or all(s not in key for s in sensitive_layer) else f.get_tensor(key).to(GET_SENSITIVE_DTYPE())) for key in f.keys()}

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        file_path = os.path.join(self.config["model_path"], f"{self.config['sub_model_folder']}/{self.config['sub_model_name']}")
        _weight_dict = self._load_safetensor_to_dict(file_path, unified_dtype, sensitive_layer)
        weight_dict = {}
        for k, v in _weight_dict.items():
            name = k[6:]
            weight = v.to(torch.bfloat16).to(self.device)
            weight_dict.update({name: weight})
        del _weight_dict
        return weight_dict

    def _init_infer_class(self):
        # update config by real model config
        with open(os.path.join(self.config["model_path"], self.config["sub_model_folder"], "config.json")) as f:
            model_config = json.load(f)
        for k in model_config.keys():
            self.config[k] = model_config[k]

        self.pre_infer_class = WanMtxg2PreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = WanMtxg2TransformerInfer
