from lightx2v.models.networks.wan.infer.animate.pre_infer import WanAnimatePreInfer
from lightx2v.models.networks.wan.infer.animate.transformer_infer import WanAnimateTransformerInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.animate.transformer_weights import WanAnimateTransformerWeights
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights


class WanAnimateModel(WanModel):
    pre_weight_class = WanPreWeights
    transformer_weight_class = WanAnimateTransformerWeights

    def __init__(self, model_path, config, device):
        self.remove_keys = ["face_encoder", "motion_encoder"]
        super().__init__(model_path, config, device)

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = WanAnimatePreInfer
        self.transformer_infer_class = WanAnimateTransformerInfer

    def set_animate_encoders(self, motion_encoder, face_encoder):
        self.pre_infer.set_animate_encoders(motion_encoder, face_encoder)
