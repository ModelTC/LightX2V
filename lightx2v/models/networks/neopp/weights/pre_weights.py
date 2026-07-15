from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import CONV2D_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class NeoppPreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.get("patch_size", 16)
        self.merge_size = config.get("merge_size", 2)
        self.mm_type = "Default"

        self.add_module(
            "vision_model_mot_gen_patch_embedding",
            CONV2D_WEIGHT_REGISTER["Default"](
                "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
                "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.bias",
                stride=self.patch_size,
            ),
        )

        self.add_module(
            "vision_model_mot_gen_dense_embedding",
            CONV2D_WEIGHT_REGISTER["Default"](
                "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight",
                "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.bias",
                stride=self.merge_size,
            ),
        )

        self.add_module(
            "timestep_embedder_mlp_0",
            MM_WEIGHT_REGISTER[self.mm_type](
                "fm_modules.timestep_embedder.mlp.0.weight",
                "fm_modules.timestep_embedder.mlp.0.bias",
            ),
        )

        self.add_module(
            "timestep_embedder_mlp_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                "fm_modules.timestep_embedder.mlp.2.weight",
                "fm_modules.timestep_embedder.mlp.2.bias",
            ),
        )

        self.add_module(
            "noise_scale_embedder_mlp_0",
            MM_WEIGHT_REGISTER[self.mm_type](
                "fm_modules.noise_scale_embedder.mlp.0.weight",
                "fm_modules.noise_scale_embedder.mlp.0.bias",
            ),
        )

        self.add_module(
            "noise_scale_embedder_mlp_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                "fm_modules.noise_scale_embedder.mlp.2.weight",
                "fm_modules.noise_scale_embedder.mlp.2.bias",
            ),
        )
