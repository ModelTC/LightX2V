from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import CONV3D_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, TENSOR_REGISTER


class DreamZeroCategoryLinearWeights(WeightModule):
    def __init__(self, prefix):
        super().__init__()
        self.add_module("W", TENSOR_REGISTER["Default"](f"{prefix}.W"))
        self.add_module("b", TENSOR_REGISTER["Default"](f"{prefix}.b"))


class DreamZeroActionEncoderWeights(WeightModule):
    def __init__(self):
        super().__init__()
        self.add_module("W1", DreamZeroCategoryLinearWeights("action_encoder.W1"))
        self.add_module("W2", DreamZeroCategoryLinearWeights("action_encoder.W2"))
        self.add_module("W3", DreamZeroCategoryLinearWeights("action_encoder.W3"))


class DreamZeroStateEncoderWeights(WeightModule):
    def __init__(self):
        super().__init__()
        self.add_module("layer1", DreamZeroCategoryLinearWeights("state_encoder.layer1"))
        self.add_module("layer2", DreamZeroCategoryLinearWeights("state_encoder.layer2"))


class DreamZeroPreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = tuple(config.get("patch_size", (1, 2, 2)))
        mm_type = config.get("dit_quant_scheme", "Default")
        if mm_type != "Default":
            assert config.get("dit_quantized") is True

        self.add_module(
            "patch_embedding",
            CONV3D_WEIGHT_REGISTER["Default"](
                "patch_embedding.weight",
                "patch_embedding.bias",
                stride=self.patch_size,
                lora_prefix="diffusion_model.patch_embedding",
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
            "proj_0",
            LN_WEIGHT_REGISTER["torch"](
                "img_emb.proj.0.weight",
                "img_emb.proj.0.bias",
                eps=1e-5,
                lora_prefix="diffusion_model.img_emb",
            ),
        )
        self.add_module(
            "proj_1",
            MM_WEIGHT_REGISTER["Default"](
                "img_emb.proj.1.weight",
                "img_emb.proj.1.bias",
                lora_prefix="diffusion_model.img_emb",
            ),
        )
        self.add_module(
            "proj_3",
            MM_WEIGHT_REGISTER["Default"](
                "img_emb.proj.3.weight",
                "img_emb.proj.3.bias",
                lora_prefix="diffusion_model.img_emb",
            ),
        )
        self.add_module(
            "proj_4",
            LN_WEIGHT_REGISTER["torch"](
                "img_emb.proj.4.weight",
                "img_emb.proj.4.bias",
                eps=1e-5,
                lora_prefix="diffusion_model.img_emb",
            ),
        )
        self.add_module("action_encoder", DreamZeroActionEncoderWeights())
        self.add_module("state_encoder", DreamZeroStateEncoderWeights())
