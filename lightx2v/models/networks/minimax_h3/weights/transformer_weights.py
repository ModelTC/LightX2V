from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


def _linear(name, bias=False):
    return MM_WEIGHT_REGISTER["Default"](f"{name}.weight", f"{name}.bias" if bias else None)


class MiniMaxH3AttentionWeights(WeightModule):
    def __init__(self, prefix, config):
        super().__init__()
        self.add_module("to_q", _linear(f"{prefix}.to_q"))
        self.add_module("to_k", _linear(f"{prefix}.to_k"))
        self.add_module("to_v", _linear(f"{prefix}.to_v"))
        qk_eps = float(config.get("qk_norm_eps", 1e-5))
        self.add_module("norm_q", RMS_WEIGHT_REGISTER["torch_native"](f"{prefix}.norm_q.weight", eps=qk_eps))
        self.add_module("norm_k", RMS_WEIGHT_REGISTER["torch_native"](f"{prefix}.norm_k.weight", eps=qk_eps))
        self.add_module("calculate", ATTN_WEIGHT_REGISTER[config.get("attn_type", "flash_attn3")]())
        self.add_module("to_out", _linear(f"{prefix}.to_out.0"))


class MiniMaxH3FeedForwardWeights(WeightModule):
    def __init__(self, prefix):
        super().__init__()
        self.add_module("in_proj", _linear(f"{prefix}.net.0.proj"))
        self.add_module("out_proj", _linear(f"{prefix}.net.2"))


class MiniMaxH3TransformerBlockWeights(WeightModule):
    def __init__(self, index, config):
        super().__init__()
        prefix = f"transformer_blocks.{index}"
        eps = float(config.get("norm_eps", 1e-5))
        self.add_module("norm1", RMS_WEIGHT_REGISTER["torch_native"](f"{prefix}.norm1.weight", eps=eps))
        self.add_module("attn", MiniMaxH3AttentionWeights(f"{prefix}.attn", config))
        self.add_module("norm2", RMS_WEIGHT_REGISTER["torch_native"](f"{prefix}.norm2.weight", eps=eps))
        self.add_module("ff", MiniMaxH3FeedForwardWeights(f"{prefix}.ff"))
        self.add_module("adaln", _linear(f"{prefix}.adaln_proj.linear", bias=True))


class MiniMaxH3TransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        if config.get("lazy_load", False):
            raise NotImplementedError(
                "MiniMax-H3 reads the official sharded checkpoint directly; per-block lazy_load requires a block-sharded checkpoint and is not supported yet. Use cpu_offload with model granularity."
            )
        self.add_module(
            "blocks",
            WeightModuleList([MiniMaxH3TransformerBlockWeights(i, config) for i in range(int(config.get("num_layers", 50)))]),
        )
