from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.utils.registry_factory import CONV3D_WEIGHT_REGISTER, EMBEDDING_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, TENSOR_REGISTER


class WanS2VPreWeights(WanPreWeights):
    def __init__(self, config):
        super().__init__(config)
        if config.get("cond_dim", 0) > 0:
            self.add_module(
                "cond_encoder",
                CONV3D_WEIGHT_REGISTER["Default"](
                    "cond_encoder.weight",
                    "cond_encoder.bias",
                    stride=self.patch_size,
                ),
            )
        self.add_module(
            "trainable_cond_mask",
            EMBEDDING_WEIGHT_REGISTER["Default"]("trainable_cond_mask.weight"),
        )
        self._register_causal_audio_encoder(config)
        if config.get("enable_framepack", False):
            self._register_frame_packer()

    def _register_causal_audio_encoder(self, config):
        enc = WeightModule()
        encoder = WeightModule()
        # key的命名语义上应该是 causal, 但是 Wan2.2 官方的 key 是 casual, 这里为了兼容性, 保留原始的 key 名称
        enc.add_module(
            "weights",
            TENSOR_REGISTER["Default"]("casual_audio_encoder.weights"),
        )
        encoder.add_module(
            "conv1_local_weight",
            TENSOR_REGISTER["Default"]("casual_audio_encoder.encoder.conv1_local.conv.weight"),
        )
        encoder.add_module(
            "conv1_local_bias",
            TENSOR_REGISTER["Default"]("casual_audio_encoder.encoder.conv1_local.conv.bias"),
        )
        if config.get("enable_adain", False):
            encoder.add_module(
                "conv1_global_weight",
                TENSOR_REGISTER["Default"]("casual_audio_encoder.encoder.conv1_global.conv.weight"),
            )
            encoder.add_module(
                "conv1_global_bias",
                TENSOR_REGISTER["Default"]("casual_audio_encoder.encoder.conv1_global.conv.bias"),
            )
            encoder.add_module(
                "final_linear",
                MM_WEIGHT_REGISTER["Default"](
                    "casual_audio_encoder.encoder.final_linear.weight",
                    "casual_audio_encoder.encoder.final_linear.bias",
                ),
            )
        encoder.add_module(
            "conv2_weight",
            TENSOR_REGISTER["Default"]("casual_audio_encoder.encoder.conv2.conv.weight"),
        )
        encoder.add_module(
            "conv2_bias",
            TENSOR_REGISTER["Default"]("casual_audio_encoder.encoder.conv2.conv.bias"),
        )
        encoder.add_module(
            "conv3_weight",
            TENSOR_REGISTER["Default"]("casual_audio_encoder.encoder.conv3.conv.weight"),
        )
        encoder.add_module(
            "conv3_bias",
            TENSOR_REGISTER["Default"]("casual_audio_encoder.encoder.conv3.conv.bias"),
        )
        encoder.add_module(
            "padding_tokens",
            TENSOR_REGISTER["Default"]("casual_audio_encoder.encoder.padding_tokens"),
        )
        enc.add_module("encoder", encoder)
        self.add_module("causal_audio_encoder", enc)

    def _register_frame_packer(self):
        fp = WeightModule()
        fp.add_module(
            "proj",
            CONV3D_WEIGHT_REGISTER["Default"](
                "frame_packer.proj.weight",
                "frame_packer.proj.bias",
                stride=(1, 2, 2),
            ),
        )
        fp.add_module(
            "proj_2x",
            CONV3D_WEIGHT_REGISTER["Default"](
                "frame_packer.proj_2x.weight",
                "frame_packer.proj_2x.bias",
                stride=(2, 4, 4),
            ),
        )
        fp.add_module(
            "proj_4x",
            CONV3D_WEIGHT_REGISTER["Default"](
                "frame_packer.proj_4x.weight",
                "frame_packer.proj_4x.bias",
                stride=(4, 8, 8),
            ),
        )
        self.add_module("frame_packer", fp)
