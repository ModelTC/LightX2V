from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.hidream_o1_image.weights.vision_weights import HidreamO1ImageVisionWeights
from lightx2v.utils.registry_factory import EMBEDDING_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class HidreamO1ImagePreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mm_type = config.get("mm_type", config.get("dit_quant_scheme", "Default"))
        self.tms_token_id = config.get("tms_token_id", 151673)
        self.frequency_embedding_size = config.get("timestep_frequency_embedding_size", 256)
        self.add_module("input_embeddings", EMBEDDING_WEIGHT_REGISTER["Default"]("model.language_model.embed_tokens.weight"))
        self.add_module("t_embedder_linear_1", self._mm("model.t_embedder1.mlp.0.weight", "model.t_embedder1.mlp.0.bias"))
        self.add_module("t_embedder_linear_2", self._mm("model.t_embedder1.mlp.2.weight", "model.t_embedder1.mlp.2.bias"))
        self.add_module("x_embedder_proj1", self._mm("model.x_embedder.proj1.weight", None))
        self.add_module("x_embedder_proj2", self._mm("model.x_embedder.proj2.weight", "model.x_embedder.proj2.bias"))
        self.add_module("visual", HidreamO1ImageVisionWeights(config))
        if config.get("_hidream_model_config") is not None:
            self.configure_model(config["_hidream_model_config"])

    def configure_model(self, model_config):
        self.model_config = model_config
        if not getattr(self.visual, "is_configured", False):
            self.visual.configure(model_config.vision_config)

    def _mm(self, weight_name, bias_name):
        return MM_WEIGHT_REGISTER[self.mm_type](weight_name, bias_name)
