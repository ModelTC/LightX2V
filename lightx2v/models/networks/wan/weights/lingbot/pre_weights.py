from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class WanLingbotPreWeights(WanPreWeights):
    """Lingbot-specific pre weights for camera/action control."""

    def __init__(self, config):
        super().__init__(config)
        mm_type = config.get("dit_quant_scheme", "Default")
        if mm_type != "Default":
            assert config.get("dit_quantized") is True

        self.add_module(
            "patch_embedding_wancamctrl",
            MM_WEIGHT_REGISTER[mm_type](
                "patch_embedding_wancamctrl.weight",
                "patch_embedding_wancamctrl.bias",
                lora_prefix="diffusion_model.patch_embedding_wancamctrl",
            ),
        )
        self.add_module(
            "c2ws_hidden_states_layer1",
            MM_WEIGHT_REGISTER[mm_type](
                "c2ws_hidden_states_layer1.weight",
                "c2ws_hidden_states_layer1.bias",
                lora_prefix="diffusion_model.c2ws_hidden_states_layer1",
            ),
        )
        self.add_module(
            "c2ws_hidden_states_layer2",
            MM_WEIGHT_REGISTER[mm_type](
                "c2ws_hidden_states_layer2.weight",
                "c2ws_hidden_states_layer2.bias",
                lora_prefix="diffusion_model.c2ws_hidden_states_layer2",
            ),
        )
