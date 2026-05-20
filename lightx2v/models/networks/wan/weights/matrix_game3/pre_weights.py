from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import (
    CONV3D_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
)


class WanMtxg3PreWeights(WeightModule):
    """Pre-processing weights for Matrix-Game-3.0.

    Handles:
    - patch_embedding (Conv3D)
    - text_embedding (2-layer MLP)
    - time_embedding + time_projection
    - patch_embedding_wancamctrl (plucker ray Linear)
    - c2ws_hidden_states_layer1/2 (camera injection MLP)
    """

    def __init__(self, config):
        super().__init__()
        self.in_dim = config["in_dim"]
        self.dim = config["dim"]
        self.patch_size = tuple(config.get("patch_size", (1, 2, 2)))
        self.config = config

        # Patch embedding
        self.add_module(
            "patch_embedding",
            CONV3D_WEIGHT_REGISTER["Default"]("patch_embedding.weight", "patch_embedding.bias", stride=self.patch_size),
        )

        # Text embedding (2-layer MLP with GELU)
        self.add_module(
            "text_embedding_0",
            MM_WEIGHT_REGISTER["Default"]("text_embedding.0.weight", "text_embedding.0.bias"),
        )
        self.add_module(
            "text_embedding_2",
            MM_WEIGHT_REGISTER["Default"]("text_embedding.2.weight", "text_embedding.2.bias"),
        )

        # Time embedding
        self.add_module(
            "time_embedding_0",
            MM_WEIGHT_REGISTER["Default"]("time_embedding.0.weight", "time_embedding.0.bias"),
        )
        self.add_module(
            "time_embedding_2",
            MM_WEIGHT_REGISTER["Default"]("time_embedding.2.weight", "time_embedding.2.bias"),
        )
        self.add_module(
            "time_projection_1",
            MM_WEIGHT_REGISTER["Default"]("time_projection.1.weight", "time_projection.1.bias"),
        )

        # Camera plucker embedding (global, before blocks)
        self.add_module(
            "patch_embedding_wancamctrl",
            MM_WEIGHT_REGISTER["Default"]("patch_embedding_wancamctrl.weight", "patch_embedding_wancamctrl.bias"),
        )
        self.add_module(
            "c2ws_hidden_states_layer1",
            MM_WEIGHT_REGISTER["Default"]("c2ws_hidden_states_layer1.weight", "c2ws_hidden_states_layer1.bias"),
        )
        self.add_module(
            "c2ws_hidden_states_layer2",
            MM_WEIGHT_REGISTER["Default"]("c2ws_hidden_states_layer2.weight", "c2ws_hidden_states_layer2.bias"),
        )
