from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.utils.registry_factory import TENSOR_REGISTER


class EdenLivePreWeights(WanPreWeights):
    def __init__(self, config):
        super().__init__(config)
        if config["use_image_encoder"]:
            self.proj_0.eps = self.proj_4.eps = 1e-5
        if config["proxy_condition_rmsnorm"]:
            self.register_parameter("condition_norm", TENSOR_REGISTER["Default"]("patch_embedding.condition_norm.weight"))
