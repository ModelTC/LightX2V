from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerWeights
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class WanS2VAudioInjectPhase(WeightModule):
    def __init__(self, injector_index, mm_type, config):
        super().__init__()
        prefix = f"audio_injector.injector.{injector_index}"
        self.add_module(
            "norm_q",
            RMS_WEIGHT_REGISTER["sgl-kernel"](f"{prefix}.norm_q.weight"),
        )
        self.add_module(
            "norm_k",
            RMS_WEIGHT_REGISTER["sgl-kernel"](f"{prefix}.norm_k.weight"),
        )
        self.add_module(
            "q",
            MM_WEIGHT_REGISTER[mm_type](f"{prefix}.q.weight", f"{prefix}.q.bias"),
        )
        self.add_module(
            "k",
            MM_WEIGHT_REGISTER[mm_type](f"{prefix}.k.weight", f"{prefix}.k.bias"),
        )
        self.add_module(
            "v",
            MM_WEIGHT_REGISTER[mm_type](f"{prefix}.v.weight", f"{prefix}.v.bias"),
        )
        self.add_module(
            "o",
            MM_WEIGHT_REGISTER[mm_type](f"{prefix}.o.weight", f"{prefix}.o.bias"),
        )
        self.add_module(
            "cross_attn_1",
            ATTN_WEIGHT_REGISTER[config.get("cross_attn_1_type", "flash_attn3")](),
        )
        if config.get("enable_adain", False):
            self.add_module(
                "adain_linear",
                MM_WEIGHT_REGISTER[mm_type](
                    f"audio_injector.injector_adain_layers.{injector_index}.linear.weight",
                    f"audio_injector.injector_adain_layers.{injector_index}.linear.bias",
                ),
            )


class WanS2VTransformerWeights(WanTransformerWeights):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__(config, lazy_load_path, lora_path)
        inject_layers = config.get("audio_inject_layers", [])
        self.injected_block_id = {layer_idx: idx for idx, layer_idx in enumerate(inject_layers)}
        for layer_idx, injector_idx in self.injected_block_id.items():
            self.blocks[layer_idx].add_module(
                "audio_inject",
                WanS2VAudioInjectPhase(injector_idx, self.mm_type, config),
            )
