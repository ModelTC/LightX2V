import torch.distributed as dist

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


def _tp_info(config):
    if not config.get("tensor_parallel", False):
        return None, 0, 1
    tp_group = config.get("device_mesh").get_group(mesh_dim="tensor_p")
    return tp_group, dist.get_rank(tp_group), dist.get_world_size(tp_group)


def _mm_weight(config, weight_name, bias_name=None, split_dim=None, create_cuda_buffer=False, create_cpu_buffer=False):
    mm_type = config.get("dit_quant_scheme", "Default")
    if config.get("tensor_parallel", False) and split_dim is not None:
        tp_group, tp_rank, tp_size = _tp_info(config)
        return MM_WEIGHT_REGISTER["TensorParallel"](
            weight_name=weight_name,
            bias_name=bias_name,
            mm_type=mm_type,
            tp_group=tp_group,
            tp_rank=tp_rank,
            tp_size=tp_size,
            split_dim=split_dim,
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=create_cpu_buffer,
        )
    return MM_WEIGHT_REGISTER[mm_type](
        weight_name,
        bias_name,
        create_cuda_buffer,
        create_cpu_buffer,
    )


def _rms_weight(config, weight_name, create_cuda_buffer=False, create_cpu_buffer=False):
    # Flux2 q/k RMSNorm weights are head_dim-sized, so TP over heads must replicate them.
    rms_norm_type = config.get("rms_norm_type", "torch")
    return RMS_WEIGHT_REGISTER[rms_norm_type](
        weight_name,
        create_cuda_buffer,
        create_cpu_buffer,
    )


class Flux2DoubleBlockWeights(WeightModule):
    """Weights for a single double-stream transformer block."""

    def __init__(self, config, block_idx, create_cuda_buffer=False, create_cpu_buffer=False):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.inner_dim = config["num_attention_heads"] * config["attention_head_dim"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.layer_norm_type = config.get("layer_norm_type", "torch")
        self.rms_norm_type = config.get("rms_norm_type", "torch")
        self.attn_type = config.get("attn_type", "flash_attn3")

        p = f"transformer_blocks.{self.block_idx}"

        self.add_module("norm1", LN_WEIGHT_REGISTER[self.layer_norm_type](eps=1e-5))
        self.add_module("norm1_context", LN_WEIGHT_REGISTER[self.layer_norm_type](eps=1e-5))
        self.add_module("norm2", LN_WEIGHT_REGISTER[self.layer_norm_type](eps=1e-5))
        self.add_module("norm2_context", LN_WEIGHT_REGISTER[self.layer_norm_type](eps=1e-5))

        self.add_module("to_q", _mm_weight(config, f"{p}.attn.to_q.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("to_k", _mm_weight(config, f"{p}.attn.to_k.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("to_v", _mm_weight(config, f"{p}.attn.to_v.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("norm_q", _rms_weight(config, f"{p}.attn.norm_q.weight", create_cuda_buffer, create_cpu_buffer))
        self.add_module("norm_k", _rms_weight(config, f"{p}.attn.norm_k.weight", create_cuda_buffer, create_cpu_buffer))

        self.add_module("add_q_proj", _mm_weight(config, f"{p}.attn.add_q_proj.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("add_k_proj", _mm_weight(config, f"{p}.attn.add_k_proj.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("add_v_proj", _mm_weight(config, f"{p}.attn.add_v_proj.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("norm_added_q", _rms_weight(config, f"{p}.attn.norm_added_q.weight", create_cuda_buffer, create_cpu_buffer))
        self.add_module("norm_added_k", _rms_weight(config, f"{p}.attn.norm_added_k.weight", create_cuda_buffer, create_cpu_buffer))

        self.add_module("to_out", _mm_weight(config, f"{p}.attn.to_out.0.weight", None, "row", create_cuda_buffer, create_cpu_buffer))
        self.add_module("to_add_out", _mm_weight(config, f"{p}.attn.to_add_out.weight", None, "row", create_cuda_buffer, create_cpu_buffer))

        self.add_module("calculate", ATTN_WEIGHT_REGISTER[self.attn_type]())

        if self.config.get("seq_parallel", False):
            self.add_module(
                "calculate_parallel",
                ATTN_WEIGHT_REGISTER[self.config["parallel"].get("seq_p_attn_type", "ulysses")](),
            )

        self.add_module("ff_net_0", _mm_weight(config, f"{p}.ff.linear_in.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("ff_net_2", _mm_weight(config, f"{p}.ff.linear_out.weight", None, "row", create_cuda_buffer, create_cpu_buffer))
        self.add_module("ff_context_net_0", _mm_weight(config, f"{p}.ff_context.linear_in.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("ff_context_net_2", _mm_weight(config, f"{p}.ff_context.linear_out.weight", None, "row", create_cuda_buffer, create_cpu_buffer))

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)


class Flux2SingleBlockWeights(WeightModule):
    """Weights for a single single-stream transformer block."""

    def __init__(self, config, block_idx, create_cuda_buffer=False, create_cpu_buffer=False):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.inner_dim = config["num_attention_heads"] * config["attention_head_dim"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.layer_norm_type = config.get("layer_norm_type", "torch")
        self.rms_norm_type = config.get("rms_norm_type", "torch")
        self.attn_type = config.get("attn_type", "flash_attn3")

        p = f"single_transformer_blocks.{self.block_idx}"

        self.add_module("norm", LN_WEIGHT_REGISTER[self.layer_norm_type](eps=1e-5))

        self.add_module("to_qkv_mlp_proj", _mm_weight(config, f"{p}.attn.to_qkv_mlp_proj.weight", None, "col", create_cuda_buffer, create_cpu_buffer))
        self.add_module("norm_q", _rms_weight(config, f"{p}.attn.norm_q.weight", create_cuda_buffer, create_cpu_buffer))
        self.add_module("norm_k", _rms_weight(config, f"{p}.attn.norm_k.weight", create_cuda_buffer, create_cpu_buffer))
        self.add_module("to_out", _mm_weight(config, f"{p}.attn.to_out.weight", None, "row", create_cuda_buffer, create_cpu_buffer))

        self.add_module("calculate", ATTN_WEIGHT_REGISTER[self.attn_type]())

        if self.config.get("seq_parallel", False):
            self.add_module(
                "calculate_parallel",
                ATTN_WEIGHT_REGISTER[self.config["parallel"].get("seq_p_attn_type", "ulysses")](),
            )

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)


class Flux2TransformerWeights(WeightModule):
    """Complete transformer weights for Flux2 model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.get("num_layers", 10)
        self.num_single_layers = config.get("num_single_layers", 20)
        self.mm_type = config.get("dit_quant_scheme", "Default")

        inner_dim = config.get("num_attention_heads", 24) * config.get("attention_head_dim", 64)

        # -- Pipeline-parallel block splitting --------------------------------
        pp_size = config.get("pipefusion_parallel", False)
        if pp_size:
            from lightx2v.common.distributed import (
                get_pipeline_parallel_rank,
                get_pipeline_parallel_world_size,
            )

            pp_rank = get_pipeline_parallel_rank()
            pp_world_size = get_pipeline_parallel_world_size()
        else:
            pp_rank = 0
            pp_world_size = 1

        if pp_world_size > 1:
            # Split double_blocks + single_blocks across pipeline stages.
            # Blocks are assigned contiguously: stage 0 gets the first chunk,
            # stage 1 the next, etc.  A stage may span the double→single
            # boundary (it will then have both types).
            total_blocks = self.num_layers + self.num_single_layers
            blocks_per_stage = (total_blocks + pp_world_size - 1) // pp_world_size
            stage_start = pp_rank * blocks_per_stage
            stage_end = min((pp_rank + 1) * blocks_per_stage, total_blocks)

            double_start = min(stage_start, self.num_layers)
            double_end = min(stage_end, self.num_layers)
            single_start = max(0, stage_start - self.num_layers)
            single_end = max(0, stage_end - self.num_layers)

            self.double_blocks = WeightModuleList([Flux2DoubleBlockWeights(config, i) for i in range(double_start, double_end)])
            self.single_blocks = WeightModuleList([Flux2SingleBlockWeights(config, i) for i in range(single_start, single_end)])
            # Track whether this stage crosses the double→single boundary
            self._has_double = double_end > double_start
            self._has_single = single_end > single_start
            self._stage_start = stage_start
            self._stage_end = stage_end
        else:
            self.double_blocks = WeightModuleList([Flux2DoubleBlockWeights(config, i) for i in range(self.num_layers)])
            self.single_blocks = WeightModuleList([Flux2SingleBlockWeights(config, i) for i in range(self.num_single_layers)])
            self._has_double = True
            self._has_single = True
            self._stage_start = 0
            self._stage_end = self.num_layers + self.num_single_layers

        self.register_offload_buffers(config)

        self.add_module("double_blocks", self.double_blocks)
        self.add_module("single_blocks", self.single_blocks)

        self.add_module("double_stream_modulation_img_linear", _mm_weight(config, "double_stream_modulation_img.linear.weight"))
        self.add_module("double_stream_modulation_txt_linear", _mm_weight(config, "double_stream_modulation_txt.linear.weight"))
        self.add_module("single_stream_modulation_linear", _mm_weight(config, "single_stream_modulation.linear.weight"))

    def register_offload_buffers(self, config):
        if config.get("cpu_offload", False) and config.get("offload_granularity", "block") == "block":
            self.offload_double_block_cuda_buffers = WeightModuleList([Flux2DoubleBlockWeights(config, i, create_cuda_buffer=True) for i in range(2)])
            self.add_module("offload_double_block_cuda_buffers", self.offload_double_block_cuda_buffers)

            self.offload_single_block_cuda_buffers = WeightModuleList([Flux2SingleBlockWeights(config, i, create_cuda_buffer=True) for i in range(2)])
            self.add_module("offload_single_block_cuda_buffers", self.offload_single_block_cuda_buffers)

    def non_block_weights_to_cuda(self, non_blocking=True):
        self.double_stream_modulation_img_linear.to_cuda(non_blocking=non_blocking)
        self.double_stream_modulation_txt_linear.to_cuda(non_blocking=non_blocking)
        self.single_stream_modulation_linear.to_cuda(non_blocking=non_blocking)

    def non_block_weights_to_cpu(self, non_blocking=True):
        self.double_stream_modulation_img_linear.to_cpu(non_blocking=non_blocking)
        self.double_stream_modulation_txt_linear.to_cpu(non_blocking=non_blocking)
        self.single_stream_modulation_linear.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for block in self.double_blocks:
            block.to_cuda(non_blocking=non_blocking)
        for block in self.single_blocks:
            block.to_cuda(non_blocking=non_blocking)
        self.non_block_weights_to_cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=True):
        for block in self.double_blocks:
            block.to_cpu(non_blocking=non_blocking)
        for block in self.single_blocks:
            block.to_cpu(non_blocking=non_blocking)
        self.non_block_weights_to_cpu(non_blocking=non_blocking)


# Backward-compatible aliases
Flux2KleinDoubleBlockWeights = Flux2DoubleBlockWeights
Flux2KleinSingleBlockWeights = Flux2SingleBlockWeights
Flux2KleinTransformerWeights = Flux2TransformerWeights
