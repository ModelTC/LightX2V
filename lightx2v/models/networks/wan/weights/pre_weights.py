from lightx2v.utils.registry_factory import (
    MM_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    CONV3D_WEIGHT_REGISTER,
)
from lightx2v.common.modules.weight_module import WeightModule


class WanPreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.in_dim = config["in_dim"]
        self.dim = config["dim"]
        self.patch_size = (1, 2, 2)
        self.config = config

        self.add_module(
            "patch_embedding",
            CONV3D_WEIGHT_REGISTER["Default"]("patch_embedding.weight", "patch_embedding.bias", stride=self.patch_size),
        )
        self.add_module(
            "text_embedding_0",
            MM_WEIGHT_REGISTER["Default"]("text_embedding.0.weight", "text_embedding.0.bias"),
        )
        self.add_module(
            "text_embedding_2",
            MM_WEIGHT_REGISTER["Default"]("text_embedding.2.weight", "text_embedding.2.bias"),
        )
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

        if config.task == "i2v":
            self.add_module(
                "proj_0",
                LN_WEIGHT_REGISTER["Default"]("img_emb.proj.0.weight", "img_emb.proj.0.bias"),
            )
            self.add_module(
                "proj_1",
                MM_WEIGHT_REGISTER["Default"]("img_emb.proj.1.weight", "img_emb.proj.1.bias"),
            )
            self.add_module(
                "proj_3",
                MM_WEIGHT_REGISTER["Default"]("img_emb.proj.3.weight", "img_emb.proj.3.bias"),
            )
            self.add_module(
                "proj_4",
                LN_WEIGHT_REGISTER["Default"]("img_emb.proj.4.weight", "img_emb.proj.4.bias"),
            )
