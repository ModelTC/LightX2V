from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class QwenImage21BlockWeights(WeightModule):
    def __init__(self, index, config):
        super().__init__()
        prefix = f"transformer_blocks.{index}"
        for name, key in {
            "q": "attn.to_q",
            "k": "attn.to_k",
            "v": "attn.to_v",
            "out": "attn.to_out.0",
            "up": "img_mlp.proj",
            "gate": "img_mlp.gate_layer",
            "down": "img_mlp.out",
        }.items():
            self.add_module(name, MM_WEIGHT_REGISTER["Default"](f"{prefix}.{key}.weight", bias_name=None))
        for name in ("q", "k"):
            self.add_module(f"norm_{name}", RMS_WEIGHT_REGISTER[config.get("rms_norm_type", "fp32_variance")](f"{prefix}.attn.norm_{name}.weight", eps=config["eps"]))
        for name in ("norm1", "norm2"):
            self.add_module(name, LN_WEIGHT_REGISTER[config.get("layer_norm_type", "torch")](eps=config["eps"]))
        self.add_module("attention", ATTN_WEIGHT_REGISTER[config.get("attn_type", "torch_sdpa")]())
        # Arbitrary triangular prefix masks use the common SDPA backend.
        self.add_module("prefix_attention", ATTN_WEIGHT_REGISTER["torch_sdpa"]())


class QwenImage21TransformerWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.add_module("blocks", WeightModuleList(QwenImage21BlockWeights(i, config) for i in range(config["num_layers"])))
