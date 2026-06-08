from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class LingbotVAPreWeights(WeightModule):
    """Pre-transformer weights actually used by LingBot-VA inference.

    LingBot-VA checkpoints may contain the original Wan conv patch embedding,
    but inference uses ``patch_embedding_mlp``. Register only the weights used by
    ``infer/lingbot_va/pre_infer.py`` so converted checkpoints without
    ``patch_embedding.weight`` still load cleanly.
    """

    def __init__(self, config):
        super().__init__()
        self.in_dim = config["in_dim"]
        self.dim = config["dim"]
        self.patch_size = tuple(config.get("patch_size", (1, 2, 2)))
        self.config = config
        mm_type = config.get("dit_quant_scheme", "Default")
        if mm_type != "Default":
            assert config.get("dit_quantized") is True

        self.add_module(
            "patch_embedding_mlp",
            MM_WEIGHT_REGISTER[mm_type](
                "patch_embedding_mlp.weight",
                "patch_embedding_mlp.bias",
                lora_prefix="diffusion_model.patch_embedding_mlp",
            ),
        )
        self.add_module(
            "action_embedder",
            MM_WEIGHT_REGISTER[mm_type](
                "action_embedder.weight",
                "action_embedder.bias",
                lora_prefix="diffusion_model.action_embedder",
            ),
        )
        self.add_module(
            "text_embedding_0",
            MM_WEIGHT_REGISTER["Default"](
                "text_embedding.0.weight",
                "text_embedding.0.bias",
                lora_prefix="diffusion_model.text_embedding",
            ),
        )
        self.add_module(
            "text_embedding_2",
            MM_WEIGHT_REGISTER["Default"](
                "text_embedding.2.weight",
                "text_embedding.2.bias",
                lora_prefix="diffusion_model.text_embedding",
            ),
        )
        self.add_module(
            "time_embedding_0",
            MM_WEIGHT_REGISTER["Default"](
                "time_embedding.0.weight",
                "time_embedding.0.bias",
                lora_prefix="diffusion_model.time_embedding",
            ),
        )
        self.add_module(
            "time_embedding_2",
            MM_WEIGHT_REGISTER["Default"](
                "time_embedding.2.weight",
                "time_embedding.2.bias",
                lora_prefix="diffusion_model.time_embedding",
            ),
        )
        self.add_module(
            "time_projection_1",
            MM_WEIGHT_REGISTER["Default"](
                "time_projection.1.weight",
                "time_projection.1.bias",
                lora_prefix="diffusion_model.time_projection",
            ),
        )
        self.add_module(
            "action_time_embedding_0",
            MM_WEIGHT_REGISTER["Default"](
                "condition_embedder_action.time_embedder.linear_1.weight",
                "condition_embedder_action.time_embedder.linear_1.bias",
                lora_prefix="diffusion_model.condition_embedder_action.time_embedder",
            ),
        )
        self.add_module(
            "action_time_embedding_2",
            MM_WEIGHT_REGISTER["Default"](
                "condition_embedder_action.time_embedder.linear_2.weight",
                "condition_embedder_action.time_embedder.linear_2.bias",
                lora_prefix="diffusion_model.condition_embedder_action.time_embedder",
            ),
        )
        self.add_module(
            "action_time_projection",
            MM_WEIGHT_REGISTER["Default"](
                "condition_embedder_action.time_proj.weight",
                "condition_embedder_action.time_proj.bias",
                lora_prefix="diffusion_model.condition_embedder_action.time_proj",
            ),
        )
