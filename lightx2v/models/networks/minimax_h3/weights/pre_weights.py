import torch.distributed as dist

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


def _linear(name, bias=False, force_fp32=False, config=None, tp_split=None):
    kind = "Default-ForceFp32" if force_fp32 else "Default"
    lora_kwargs = {"lora_prefix": "token_refiner"} if name.startswith("token_refiner.") else {}
    if config is not None and config.get("tensor_parallel", False) and tp_split is not None:
        tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
        tp_mm_type = config.get("tp_mm_type", "TensorParallel")
        return MM_WEIGHT_REGISTER[tp_mm_type](
            weight_name=f"{name}.weight",
            bias_name=f"{name}.bias" if bias else None,
            mm_type=kind,
            tp_group=tp_group,
            tp_rank=dist.get_rank(tp_group),
            tp_size=dist.get_world_size(tp_group),
            split_dim=tp_split,
            lora_column_chunks=2 if ".ff.net.0.proj" in name else 1,
            **lora_kwargs,
        )
    return MM_WEIGHT_REGISTER[kind](f"{name}.weight", f"{name}.bias" if bias else None, **lora_kwargs)


def _rms(config, name, eps):
    return RMS_WEIGHT_REGISTER[config.get("rms_type", "torch_native")](name, eps=eps)


class MiniMaxH3RefinerAttentionWeights(WeightModule):
    def __init__(self, prefix, config):
        super().__init__()
        self.add_module("to_q", _linear(f"{prefix}.to_q", config=config, tp_split="col"))
        self.add_module("to_k", _linear(f"{prefix}.to_k", config=config, tp_split="col"))
        self.add_module("to_v", _linear(f"{prefix}.to_v", config=config, tp_split="col"))
        self.add_module(
            "norm_q",
            _rms(config, f"{prefix}.norm_q.weight", eps=float(config.get("qk_norm_eps", 1e-5))),
        )
        self.add_module(
            "norm_k",
            _rms(config, f"{prefix}.norm_k.weight", eps=float(config.get("qk_norm_eps", 1e-5))),
        )
        # H3's text refiner attends over a short text-only sequence, while the
        # main transformer attends over the much longer packed AV sequence.
        # Allow sparse main attention without paying its setup/quality cost in
        # the refiner. Existing configs retain their previous shared backend.
        attn_type = config.get("refiner_attn_type", config.get("attn_type", "flash_attn3"))
        attention_cls = ATTN_WEIGHT_REGISTER[attn_type]
        if attn_type == "dynamic_sparse_attn":
            calculate = attention_cls(config.get("dynamic_sparse_attn_setting", {}))
        else:
            calculate = attention_cls()
        if attn_type == "sol_attn":
            calculate.set_config(config.get("sol_attn_setting", {}))
        self.add_module("calculate", calculate)
        self.add_module("to_out", _linear(f"{prefix}.to_out.0", config=config, tp_split="row"))


class MiniMaxH3FeedForwardWeights(WeightModule):
    def __init__(self, prefix, config):
        super().__init__()
        self.add_module("in_proj", _linear(f"{prefix}.net.0.proj", config=config, tp_split="col"))
        self.add_module("out_proj", _linear(f"{prefix}.net.2", config=config, tp_split="row"))


class MiniMaxH3TokenRefinerBlockWeights(WeightModule):
    def __init__(self, index, config):
        super().__init__()
        prefix = f"token_refiner.refiner_blocks.{index}"
        eps = float(config.get("norm_eps", 1e-5))
        self.add_module("norm1", _rms(config, f"{prefix}.norm1.weight", eps=eps))
        self.add_module("attn", MiniMaxH3RefinerAttentionWeights(f"{prefix}.attn", config))
        self.add_module("norm2", _rms(config, f"{prefix}.norm2.weight", eps=eps))
        self.add_module("ff", MiniMaxH3FeedForwardWeights(f"{prefix}.ff", config))


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
