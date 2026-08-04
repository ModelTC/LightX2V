from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


def _linear(name, bias=False, force_fp32=False):
    kind = "Default-ForceFp32" if force_fp32 else "Default"
    return MM_WEIGHT_REGISTER[kind](f"{name}.weight", f"{name}.bias" if bias else None)


def _rms(config, name, eps):
    return RMS_WEIGHT_REGISTER[config.get("rms_type", "torch_native")](name, eps=eps)


class MiniMaxH3RefinerAttentionWeights(WeightModule):
    def __init__(self, prefix, config):
        super().__init__()
        self.add_module("to_q", _linear(f"{prefix}.to_q"))
        self.add_module("to_k", _linear(f"{prefix}.to_k"))
        self.add_module("to_v", _linear(f"{prefix}.to_v"))
        self.add_module(
            "norm_q",
            _rms(config, f"{prefix}.norm_q.weight", eps=float(config.get("qk_norm_eps", 1e-5))),
        )
        self.add_module(
            "norm_k",
            _rms(config, f"{prefix}.norm_k.weight", eps=float(config.get("qk_norm_eps", 1e-5))),
        )
        self.add_module("calculate", ATTN_WEIGHT_REGISTER[config.get("attn_type", "flash_attn3")]())
        self.add_module("to_out", _linear(f"{prefix}.to_out.0"))


class MiniMaxH3FeedForwardWeights(WeightModule):
    def __init__(self, prefix):
        super().__init__()
        self.add_module("in_proj", _linear(f"{prefix}.net.0.proj"))
        self.add_module("out_proj", _linear(f"{prefix}.net.2"))


class MiniMaxH3TokenRefinerBlockWeights(WeightModule):
    def __init__(self, index, config):
        super().__init__()
        prefix = f"token_refiner.refiner_blocks.{index}"
        eps = float(config.get("norm_eps", 1e-5))
        self.add_module("norm1", _rms(config, f"{prefix}.norm1.weight", eps=eps))
        self.add_module("attn", MiniMaxH3RefinerAttentionWeights(f"{prefix}.attn", config))
        self.add_module("norm2", _rms(config, f"{prefix}.norm2.weight", eps=eps))
        self.add_module("ff", MiniMaxH3FeedForwardWeights(f"{prefix}.ff"))


class MiniMaxH3PreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        # The released checkpoint deliberately keeps the two media projections
        # and timestep MLP in fp32.  The text projection/refiner stay bf16.
        self.add_module("proj_in", _linear("proj_in", bias=True, force_fp32=True))
        self.add_module("audio_proj_in", _linear("audio_proj_in", bias=True, force_fp32=True))
        self.add_module("context_embedder", _linear("context_embedder", bias=True))
        self.add_module("time_linear_1", _linear("time_embedder.linear_1", bias=True, force_fp32=True))
        self.add_module("time_linear_2", _linear("time_embedder.linear_2", bias=True, force_fp32=True))
        self.add_module(
            "refiner_blocks",
            WeightModuleList([MiniMaxH3TokenRefinerBlockWeights(i, config) for i in range(int(config.get("num_refiner_layers", 2)))]),
        )
        self.add_module(
            "refiner_final_norm",
            _rms(config, "token_refiner.final_norm.weight", eps=float(config.get("final_norm_eps", 1e-5))),
        )
