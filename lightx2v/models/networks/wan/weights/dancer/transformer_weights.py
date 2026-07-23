from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerWeights
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, TENSOR_REGISTER


class WanDancerInjectorWeights(WeightModule):
    """Only V/O are needed: attention over one audio token has softmax == 1."""

    def __init__(self, injector_id):
        super().__init__()
        prefix = f"music_injector.injector.{injector_id}"
        self.add_module("v", MM_WEIGHT_REGISTER["Default"](f"{prefix}.v.weight", f"{prefix}.v.bias"))
        self.add_module("o", MM_WEIGHT_REGISTER["Default"](f"{prefix}.o.weight", f"{prefix}.o.bias"))


class WanDancerTransformerWeights(WanTransformerWeights):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        if config.get("lazy_load", False):
            raise NotImplementedError("Wan-Dancer currently supports model/block offload, not disk lazy-load.")
        super().__init__(config, lazy_load_path=lazy_load_path, lora_path=lora_path)

        self.add_module("head_global", MM_WEIGHT_REGISTER["Default"]("head_global.head.weight", "head_global.head.bias"))
        self.register_parameter("head_global_modulation", TENSOR_REGISTER["Default"]("head_global.modulation"))
        self.music_injectors = WeightModuleList([WanDancerInjectorWeights(i) for i in range(len(config["music_inject_layers"]))])
        self.add_module("music_injectors", self.music_injectors)

    def non_block_weights_to_cuda(self):
        super().non_block_weights_to_cuda()
        self.head_global.to_cuda()
        self.head_global_modulation.to_cuda()
        self.music_injectors.to_cuda()

    def non_block_weights_to_cpu(self):
        super().non_block_weights_to_cpu()
        self.head_global.to_cpu()
        self.head_global_modulation.to_cpu()
        self.music_injectors.to_cpu()
