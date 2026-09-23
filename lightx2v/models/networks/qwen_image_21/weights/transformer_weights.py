import torch.distributed as dist

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.ops.mm.mm_weight import unwrap_tp_weight
from lightx2v.models.networks.qwen_image_21.fp8_f16_accum_policy import ACTIVATION_QMAX
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class QwenImage21BlockWeights(WeightModule):
    def __init__(self, index, config, create_cuda_buffer=False):
        super().__init__()
        prefix = f"transformer_blocks.{index}"
        mm_type = config["dit_quant_scheme"] if config.get("dit_quantized", False) else "Default"
        tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p") if config.get("tensor_parallel", False) else None
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        for name, (key, tp_split) in {
            "q": ("attn.to_q", "col"),
            "k": ("attn.to_k", "col"),
            "v": ("attn.to_v", "col"),
            "out": ("attn.to_out.0", "row"),
            "up": ("img_mlp.proj", "col"),
            "gate": ("img_mlp.gate_layer", "col"),
            "down": ("img_mlp.out", "row"),
        }.items():
            if tp_group is not None:
                tp_mm_type = config.get("tp_mm_type", "TensorParallel")
                linear = MM_WEIGHT_REGISTER[tp_mm_type](
                    f"{prefix}.{key}.weight",
                    bias_name=None,
                    mm_type=mm_type,
                    tp_group=tp_group,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    split_dim=tp_split,
                    create_cuda_buffer=create_cuda_buffer,
                )
            else:
                linear = MM_WEIGHT_REGISTER[mm_type](
                    f"{prefix}.{key}.weight",
                    bias_name=None,
                    create_cuda_buffer=create_cuda_buffer,
                )
            if mm_type == "fp8-f16-accum":
                unwrap_tp_weight(linear).enable_fp8_f16_accum(config.get("dit_fp8_activation_qmax", ACTIVATION_QMAX))
            self.add_module(name, linear)
        for name in ("q", "k"):
            self.add_module(
                f"norm_{name}",
                RMS_WEIGHT_REGISTER[config.get("rms_norm_type", "fp32_variance")](
                    f"{prefix}.attn.norm_{name}.weight",
                    create_cuda_buffer=create_cuda_buffer,
                    eps=config["eps"],
                ),
            )
        for name in ("norm1", "norm2"):
            self.add_module(name, LN_WEIGHT_REGISTER[config.get("layer_norm_type", "torch")](eps=config["eps"]))
        attn_type = config.get("attn_type", "torch_sdpa")
        attention_cls = ATTN_WEIGHT_REGISTER[attn_type]
        if attn_type == "dynamic_sparse_attn":
            attention = attention_cls(config.get("dynamic_sparse_attn_setting", {}))
        else:
            attention = attention_cls()
        self.add_module("attention", attention)
        if config.get("seq_parallel", False):
            parallel = config["parallel"]
            self.add_module("calculate_parallel", ATTN_WEIGHT_REGISTER[parallel.get("seq_p_attn_type", "ulysses")]())
        # Arbitrary triangular prefix masks use the common SDPA backend.
        self.add_module("prefix_attention", ATTN_WEIGHT_REGISTER["torch_sdpa"]())


class QwenImage21TransformerWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        blocks = WeightModuleList(QwenImage21BlockWeights(i, config) for i in range(config["num_layers"]))
        if config.get("cpu_offload", False) and config.get("offload_granularity", "model") == "block":
            self.offload_block_cuda_buffers = WeightModuleList(QwenImage21BlockWeights(i, config, create_cuda_buffer=True) for i in range(2))
            # Register device buffers before CPU source blocks so buffer
            # allocation can use the checkpoint metadata.
            self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)
            self.offload_phase_cuda_buffers = None
        self.add_module("blocks", blocks)
