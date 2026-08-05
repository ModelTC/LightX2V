import torch
import torch.distributed as dist

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.minimax_h3.infer.triton_ops import MiniMaxH3TritonRope  # noqa: F401
from lightx2v.models.networks.minimax_h3.weights.tensor_parallel import MiniMaxH3TensorParallelLinear
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, ROPE_REGISTER


def _linear(config, name, bias=False, create_cuda_buffer=False, tp_split=None):
    if config.get("tensor_parallel", False) and tp_split is not None:
        tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
        return MiniMaxH3TensorParallelLinear(
            weight_name=f"{name}.weight",
            bias_name=f"{name}.bias" if bias else None,
            mm_type=config.get("dit_quant_scheme", "Default"),
            tp_group=tp_group,
            tp_rank=dist.get_rank(tp_group),
            tp_size=dist.get_world_size(tp_group),
            split_dim=tp_split,
            create_cuda_buffer=create_cuda_buffer,
        )
    return MM_WEIGHT_REGISTER[config.get("dit_quant_scheme", "Default")](
        f"{name}.weight",
        f"{name}.bias" if bias else None,
        create_cuda_buffer=create_cuda_buffer,
    )


def _rms(config, name, eps, create_cuda_buffer=False):
    return RMS_WEIGHT_REGISTER[config.get("rms_type", "torch_native")](
        name,
        create_cuda_buffer=create_cuda_buffer,
        eps=eps,
    )


class MiniMaxH3AttentionWeights(WeightModule):
    def __init__(self, prefix, config, create_cuda_buffer=False):
        super().__init__()
        self.add_module("to_q", _linear(config, f"{prefix}.to_q", create_cuda_buffer=create_cuda_buffer, tp_split="col"))
        self.add_module("to_k", _linear(config, f"{prefix}.to_k", create_cuda_buffer=create_cuda_buffer, tp_split="col"))
        self.add_module("to_v", _linear(config, f"{prefix}.to_v", create_cuda_buffer=create_cuda_buffer, tp_split="col"))
        qk_eps = float(config.get("qk_norm_eps", 1e-5))
        self.add_module(
            "norm_q",
            _rms(
                config,
                f"{prefix}.norm_q.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=qk_eps,
            ),
        )
        self.add_module(
            "norm_k",
            _rms(
                config,
                f"{prefix}.norm_k.weight",
                create_cuda_buffer=create_cuda_buffer,
                eps=qk_eps,
            ),
        )
        self.add_module(
            "rope",
            ROPE_REGISTER[config.get("rope_type", "torch_real_rope")](
                layout="split_half",
                compute_dtype=torch.float32,
            ),
        )
        self.add_module("calculate", ATTN_WEIGHT_REGISTER[config.get("attn_type", "flash_attn3")]())
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
        self.add_module("in_proj", _linear(config, f"{prefix}.net.0.proj", create_cuda_buffer=create_cuda_buffer, tp_split="col"))
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
        # AdaLN is the largest per-block projection in H3.  Its output is
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
            # Register device buffers before source blocks: buffer allocation
            # needs checkpoint metadata that normal CPU loading consumes.
            self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)
            self.offload_phase_cuda_buffers = None
        self.add_module("blocks", self.blocks)
