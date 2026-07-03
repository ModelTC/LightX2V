from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import CONV3D_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class FastWAMExpertPreWeights(WeightModule):
    def __init__(self, prefix, *, has_patch_embedding=False, has_action_encoder=False):
        super().__init__()
        if has_patch_embedding:
            self.add_module(
                "patch_embedding",
                CONV3D_WEIGHT_REGISTER["Default"](
                    f"{prefix}.patch_embedding.weight",
                    f"{prefix}.patch_embedding.bias",
                    stride=(1, 2, 2),
                ),
            )
        if has_action_encoder:
            self.add_module(
                "action_encoder",
                MM_WEIGHT_REGISTER["Default"](
                    f"{prefix}.action_encoder.weight",
                    f"{prefix}.action_encoder.bias",
                ),
            )

        self.add_module(
            "text_embedding_0",
            MM_WEIGHT_REGISTER["Default"](
                f"{prefix}.text_embedding.0.weight",
                f"{prefix}.text_embedding.0.bias",
            ),
        )
        self.add_module(
            "text_embedding_2",
            MM_WEIGHT_REGISTER["Default"](
                f"{prefix}.text_embedding.2.weight",
                f"{prefix}.text_embedding.2.bias",
            ),
        )
        self.add_module(
            "time_embedding_0",
            MM_WEIGHT_REGISTER["Default"](
                f"{prefix}.time_embedding.0.weight",
                f"{prefix}.time_embedding.0.bias",
            ),
        )
        self.add_module(
            "time_embedding_2",
            MM_WEIGHT_REGISTER["Default"](
                f"{prefix}.time_embedding.2.weight",
                f"{prefix}.time_embedding.2.bias",
            ),
        )
        self.add_module(
            "time_projection_1",
            MM_WEIGHT_REGISTER["Default"](
                f"{prefix}.time_projection.1.weight",
                f"{prefix}.time_projection.1.bias",
            ),
        )


class FastWAMPreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add_module(
            "video",
            FastWAMExpertPreWeights(
                "mixtures.video",
                has_patch_embedding=True,
            ),
        )
        self.add_module(
            "action",
            FastWAMExpertPreWeights(
                "mixtures.action",
                has_action_encoder=True,
            ),
        )
        self.add_module(
            "proprio_encoder",
            MM_WEIGHT_REGISTER["Default"](
                "proprio_encoder.weight",
                "proprio_encoder.bias",
            ),
        )
