import torch
import torch.distributed as dist

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, ROPE_REGISTER


def _ensure_h3_leaf_weights_registered():
    from lightx2v.common.ops.rope import MiniMaxH3SGLRope  # noqa: F401
    from lightx2v.models.networks.minimax_h3.weights.merged_qkv import MiniMaxH3SGLMergedQKVWeight  # noqa: F401
    from lightx2v.models.networks.minimax_h3.weights.qk_norm import MiniMaxH3SGLQKRMSNorm  # noqa: F401
    from lightx2v.models.networks.minimax_h3.weights.reordered_mlp import MiniMaxH3SGLReorderedMLPWeight  # noqa: F401


def _linear(config, name, bias=False, create_cuda_buffer=False, tp_split=None):
    lora_prefix = "transformer_blocks"
    if config.get("tensor_parallel", False) and tp_split is not None:
        tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
        tp_mm_type = config.get("tp_mm_type", "TensorParallel")
        return MM_WEIGHT_REGISTER[tp_mm_type](
            weight_name=f"{name}.weight",
            bias_name=f"{name}.bias" if bias else None,
            mm_type=config.get("dit_quant_scheme", "Default"),
            tp_group=tp_group,
            tp_rank=dist.get_rank(tp_group),
            tp_size=dist.get_world_size(tp_group),
            split_dim=tp_split,
            lora_column_chunks=2 if ".ff.net.0.proj" in name else 1,
            create_cuda_buffer=create_cuda_buffer,
            lora_prefix=lora_prefix,
        )
    return MM_WEIGHT_REGISTER[config.get("dit_quant_scheme", "Default")](
        f"{name}.weight",
        f"{name}.bias" if bias else None,
        create_cuda_buffer=create_cuda_buffer,
        lora_prefix=lora_prefix,
    )


def _packed_linear_kwargs(config):
    kwargs = {
        "mm_type": config.get("dit_quant_scheme", "Default"),
        "tp_group": None,
        "tp_rank": 0,
        "tp_size": 1,
        "config": config,
    }
    if config.get("tensor_parallel", False):
        group = config["device_mesh"].get_group(mesh_dim="tensor_p")
        kwargs.update(
            tp_group=group,
            tp_rank=dist.get_rank(group),
            tp_size=dist.get_world_size(group),
        )
    return kwargs


def _rms(config, name, eps, create_cuda_buffer=False, kind=None):
    return RMS_WEIGHT_REGISTER[kind or config.get("rms_type", "torch_native")](
        name,
        create_cuda_buffer=create_cuda_buffer,
        eps=eps,
    )


class MiniMaxH3AttentionWeights(WeightModule):
    def __init__(self, prefix, config, create_cuda_buffer=False):
        super().__init__()
        _ensure_h3_leaf_weights_registered()
        self.add_module(
            "qkv",
            MM_WEIGHT_REGISTER["h3ref_sgl_merged_qkv"](
                weight_names=tuple(f"{prefix}.to_{name}.weight" for name in ("q", "k", "v")),
                create_cuda_buffer=create_cuda_buffer,
                **_packed_linear_kwargs(config),
            ),
        )

        qk_eps = float(config.get("qk_norm_eps", 1e-5))
        qk_norm_kind = "h3ref_sgl_qk_rms_norm"
        self.add_module(
            "norm_q",
            _rms(
                config,
                f"{prefix}.norm_q.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=qk_eps,
                kind=qk_norm_kind,
            ),
        )
        self.add_module(
            "norm_k",
            _rms(
                config,
                f"{prefix}.norm_k.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=qk_eps,
                kind=qk_norm_kind,
            ),
        )
        rope_kind = config.get("rope_type", "h3_sgl_rope")
        self.add_module(
            "rope",
            ROPE_REGISTER[rope_kind](
                layout="split_half",
                compute_dtype=torch.bfloat16,
            ),
        )
        attn_type = config.get("attn_type", "flash_attn3")
        attention_cls = ATTN_WEIGHT_REGISTER[attn_type]
        if attn_type == "dynamic_sparse_attn":
            calculate = attention_cls(config.get("dynamic_sparse_attn_setting", {}))
        else:
            calculate = attention_cls()
        if attn_type == "sol_attn":
            calculate.set_config(config.get("sol_attn_setting", {}))
        self.add_module("calculate", calculate)
        if config.get("seq_parallel", False):
            parallel = config.get("parallel", {})
            self.add_module(
                "calculate_parallel",
                ATTN_WEIGHT_REGISTER[parallel.get("seq_p_attn_type", "ulysses")](a2a_backend=parallel.get("seq_p_a2a_backend", "torch")),
            )
        self.add_module("to_out", _linear(config, f"{prefix}.to_out.0", create_cuda_buffer=create_cuda_buffer, tp_split="row"))


class MiniMaxH3FeedForwardWeights(WeightModule):
    def __init__(self, prefix, config, create_cuda_buffer=False):
        super().__init__()
        _ensure_h3_leaf_weights_registered()
        in_proj = MM_WEIGHT_REGISTER["h3ref_sgl_reordered_mlp"](
            weight_name=f"{prefix}.net.0.proj.weight",
            create_cuda_buffer=create_cuda_buffer,
            lora_prefix="transformer_blocks",
            **_packed_linear_kwargs(config),
        )
        self.add_module("in_proj", in_proj)
        self.add_module("out_proj", _linear(config, f"{prefix}.net.2", create_cuda_buffer=create_cuda_buffer, tp_split="row"))


class MiniMaxH3TransformerBlockWeights(WeightModule):
    def __init__(self, index, config, create_cuda_buffer=False):
        super().__init__()
        prefix = f"transformer_blocks.{index}"
        eps = float(config.get("norm_eps", 1e-5))
        self.add_module(
            "norm1",
            _rms(
                config,
                f"{prefix}.norm1.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=eps,
            ),
        )
        self.add_module("attn", MiniMaxH3AttentionWeights(f"{prefix}.attn", config, create_cuda_buffer))
        self.add_module(
            "norm2",
            _rms(
                config,
                f"{prefix}.norm2.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=eps,
            ),
        )
        self.add_module("ff", MiniMaxH3FeedForwardWeights(f"{prefix}.ff", config, create_cuda_buffer))
        if not config.get("use_adaln_cache", False):
            # ADALN CACHE SYNC: The offline builder reads this key and mirrors
            # the unquantized projection; update the offline builder if it changes.
            # AdaLN is the largest per-block projection in H3. Its output is
            # column-sharded here and gathered once per block before modulation.
            self.add_module("adaln", _linear(config, f"{prefix}.adaln_proj.linear", bias=True, create_cuda_buffer=create_cuda_buffer, tp_split="col"))


class MiniMaxH3TransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        if config.get("lazy_load", False):
            raise NotImplementedError(
                "MiniMax-H3 reads the official sharded checkpoint directly; disk lazy_load requires a converted block-sharded checkpoint and is not supported yet. Use lazy_load=false with model or block CPU offload."
            )
        self.blocks = WeightModuleList([MiniMaxH3TransformerBlockWeights(i, config) for i in range(int(config.get("num_layers", 50)))])
        if config.get("cpu_offload", False) and config.get("offload_granularity", "model") == "block":
            self.offload_block_cuda_buffers = WeightModuleList([MiniMaxH3TransformerBlockWeights(i, config, create_cuda_buffer=True) for i in range(2)])
            self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)
            self.offload_phase_cuda_buffers = None
        self.add_module("blocks", self.blocks)
