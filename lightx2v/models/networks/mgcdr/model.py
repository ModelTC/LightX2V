import torch
from pydantic import BaseModel
from lightx2v.models.schedulers.mgcdr.scheduler import MagicDriverScheduler
from lightx2v.models.networks.mgcdr.infer.transformer_infer import MagicDriveTransformerInfer
from lightx2v.models.networks.mgcdr.infer.pre_infer import MagicDrivePreInfer
from lightx2v.models.networks.mgcdr.infer.post_infer import MagicDrivePostInfer
from lightx2v.models.networks.mgcdr.weights.transformer_weights import MagicDriveTransformerWeight
from lightx2v.models.networks.mgcdr.weights.pre_weights import MagicDrivePreWeights
from lightx2v.models.networks.mgcdr.weights.post_weights import MagicDrivePostWeights
from lightx2v.models.networks.mgcdr.infer.feature_caching.transformer_infer import (
    MagicDriveTransformerInferTeaCaching,
)


# class ModelInputs(BaseModel):
#     y: torch.Tensor
#     mask: torch.Tensor
#     maps: torch.Tensor
#     bbox: torch.Tensor
#     cams: torch.Tensor
#     rel_pos: torch.Tensor
#     fps: torch.Tensor
#     drop_cond_mask: torch.Tensor


class MagicDriveModel:
    pre_weight_class = MagicDrivePreWeights
    post_weight_class = MagicDrivePostWeights
    transformer_weight_class = MagicDriveTransformerWeight
    
    def __init__(self, config, device):
        self.config = config
        self.model_path = config.get('model_path')
        self.device = device
        self._init_infer_class()
        self._init_weights()
        self._init_infer()
        # self.set_scheduler()
        
    def _init_infer_class(self):
        self.pre_infer_class = MagicDrivePreInfer
        self.post_infer_class = MagicDrivePostInfer
        self.transformer_infer_class = MagicDriveTransformerInfer

    def _load_ckpt(self):
        weight_dict = torch.load(self.model_path, map_location=self.device)
        for k, v in weight_dict.items():
            weight_dict[k] = v.bfloat16()
        return weight_dict
    
    def _init_weights(self):
        weight_dict = self._load_ckpt()
        self.pre_weight = self.pre_weight_class(self.config)
        self.post_weight = self.post_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        
        self.pre_weight.load(weight_dict)
        self.post_weight.load(weight_dict)
        self.transformer_weights.load(weight_dict)
    
    def _init_infer_class(self):
        self.pre_infer_class = MagicDrivePreInfer
        self.post_infer_class = MagicDrivePostInfer
        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = MagicDriveTransformerInfer
        elif self.config["feature_caching"] == "Tea":
            self.transformer_infer_class = MagicDriveTransformerInferTeaCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")

    def set_scheduler(self, scheduler: MagicDriverScheduler):
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.post_infer.set_scheduler(scheduler)
        self.transformer_infer.set_scheduler(scheduler)

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        
    @torch.no_grad()
    def infer(self, inputs: dict):
        # import pdb; pdb.set_trace()
        x, y, c, t, t_mlp, y_lens, x_mask, t0, t0_mlp, T, H, W, S, NC, Tx, Hx, Wx, mv_order_map  = self.pre_infer.infer(self.pre_weight, **inputs)
        x = self.transformer_infer.infer(self.transformer_weights, x, y, c, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map)
        x = self.post_infer.infer(self.post_weight, x, t, x_mask, t0, S, NC, T, H, W, Tx, Hx, Wx)
        inputs['x'] = x
        # return x
        
