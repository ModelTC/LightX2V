from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import (
    MM_WEIGHT_REGISTER,
    EMBEDDING_WEIGHT_REGISTER
)


class Qwen2PreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # connector
        self.add_module(
            "fc1",
            MM_WEIGHT_REGISTER["Default"]("connector.fc1.weight", "connector.fc1.bias"),
        )
        self.add_module(
            "fc2",
            MM_WEIGHT_REGISTER["Default"]("connector.fc2.weight", "connector.fc2.bias"),
        )
        # language_model
        self.add_module(
            "lm_head",
            MM_WEIGHT_REGISTER["Default"]("language_model.lm_head.weight"),
        )
        self.add_module(
            "embed_tokens",
            EMBEDDING_WEIGHT_REGISTER["Default"]("language_model.model.embed_tokens.weight"),
        )
        # vae2llm
        self.add_module(
            "vae2llm",
            MM_WEIGHT_REGISTER["Default"]("vae2llm.weight", "vae2llm.bias"),
        )

        # time_embedder
        self.add_module(
            "mlp_0",
            MM_WEIGHT_REGISTER["Default"]("time_embedder.mlp.0.weight", "time_embedder.mlp.0.bias"),
        )
        self.add_module(
            "mlp_2",
            MM_WEIGHT_REGISTER["Default"]("time_embedder.mlp.2.weight", "time_embedder.mlp.2.bias"),
        )