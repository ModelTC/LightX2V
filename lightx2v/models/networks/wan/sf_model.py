import torch

from lightx2v.models.networks.wan.autoregressive_model import WanAutoregressiveModel
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.self_forcing.pre_infer import WanSFPreInfer
from lightx2v.models.networks.wan.infer.self_forcing.transformer_infer import WanSFTransformerInfer
from lightx2v.utils.envs import GET_DTYPE


class WanSFModel(WanAutoregressiveModel):
    def _load_ckpt(self, unified_dtype, sensitive_layer):
        file_path = self.config["dit_original_ckpt"]
        if file_path.endswith(".safetensors"):
            _weight_dict = super()._load_ckpt(unified_dtype, sensitive_layer)
        else:
            _weight_dict = torch.load(file_path, map_location="cpu")
            _weight_dict = _weight_dict.get("generator_ema", _weight_dict)
        # TP weight leaves slice on the CPU before transferring their rank's
        # shard; moving the full checkpoint here defeats that loading policy.
        load_device = torch.device("cpu") if self.use_tp else self.device
        weight_dict = {}
        for k, v in _weight_dict.items():
            name = k[6:]
            weight = v.to(dtype=GET_DTYPE(), device=load_device)
            weight_dict.update({name: weight})
        del _weight_dict
        return weight_dict

    def _load_quant_ckpt(self, unified_dtype, sensitive_layer):
        _weight_dict = super()._load_quant_ckpt(unified_dtype, sensitive_layer)
        weight_dict = {}
        for k, v in _weight_dict.items():
            weight_dict.update({k[6:]: v})
        del _weight_dict
        return weight_dict

    def _init_infer_class(self):
        self.pre_infer_class = WanSFPreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = WanSFTransformerInfer
