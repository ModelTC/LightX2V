import os

import torch

from lightx2v.models.networks.wan.infer.matrix_game2.pre_infer import WanMtxg2PreInfer
from lightx2v.models.networks.wan.infer.matrix_game2.transformer_infer import WanMtxg2TransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.sf_model import WanSFModel
from lightx2v.models.networks.wan.weights.matrix_game2.pre_weights import WanMtxg2PreWeights
from lightx2v.models.networks.wan.weights.matrix_game2.transformer_weights import WanActionTransformerWeights


class WanSFMtxg2Model(WanSFModel):
    pre_weight_class = WanMtxg2PreWeights
    transformer_weight_class = WanActionTransformerWeights

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        file_path = os.path.join(self.config["model_path"], self.config["sub_model_folder"], self.config["sub_model_name"])
        weight_dict = self._load_safetensor_to_dict(file_path, unified_dtype, sensitive_layer)
        return {key[6:]: weight.to(torch.bfloat16) for key, weight in weight_dict.items()}

    def _init_infer_class(self):
        self.pre_infer_class = WanMtxg2PreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = WanMtxg2TransformerInfer
