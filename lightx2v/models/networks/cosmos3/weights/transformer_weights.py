import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.ops.rope import build_rope_weight
from lightx2v.models.networks.cosmos3.infer.utils import Cosmos3Rope  # noqa: F401
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class Cosmos3TransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        self.config = config
        self.layers_num = config["num_hidden_layers"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.rms_norm_type = config.get("rms_norm_type", "one-pass")
        self.attn_rms_norm_type = config.get("attn_rms_norm_type", self.rms_norm_type)
        self.lazy_load = config.get("lazy_load", False)
        if self.lazy_load:
            self.lazy_load_file = lazy_load_path
        else:
            self.lazy_load_file = None
        layers = WeightModuleList(
            Cosmos3TransformerLayerWeights(
                layer_idx,
                self.mm_type,
                config,
                self.rms_norm_type,
                self.attn_rms_norm_type,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_path=lora_path,
            )
            for layer_idx in range(self.layers_num)
        )
        self.register_offload_buffers(config, lazy_load_path, lora_path)
        self.add_module("layers", layers)

    def register_offload_buffers(self, config, lazy_load_path, lora_path):
        if not config.get("cpu_offload", False):
            return
        if config.get("offload_granularity", "block") != "block":
            raise NotImplementedError("Cosmos3 transformer supports only block-level cpu_offload.")

        self.offload_blocks_num = 2
        self.offload_block_cuda_buffers = WeightModuleList(
            [
                Cosmos3TransformerLayerWeights(
                    layer_idx=i,
                    mm_type=self.mm_type,
                    config=config,
                    rms_norm_type=self.rms_norm_type,
                    attn_rms_norm_type=self.attn_rms_norm_type,
                    create_cuda_buffer=True,
                    create_cpu_buffer=False,
                    lazy_load=self.lazy_load,
                    lazy_load_file=lazy_load_path,
                    lora_path=lora_path,
                )
                for i in range(self.offload_blocks_num)
            ]
        )
        self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)
        self.offload_phase_cuda_buffers = None

        if self.lazy_load:
            self.offload_block_cpu_buffers = WeightModuleList(
                [
                    Cosmos3TransformerLayerWeights(
                        layer_idx=i,
                        mm_type=self.mm_type,
                        config=config,
                        rms_norm_type=self.rms_norm_type,
                        attn_rms_norm_type=self.attn_rms_norm_type,
                        create_cuda_buffer=False,
                        create_cpu_buffer=True,
                        lazy_load=self.lazy_load,
                        lazy_load_file=lazy_load_path,
                        lora_path=lora_path,
                    )
                    for i in range(self.offload_blocks_num)
                ]
            )
            self.add_module("offload_block_cpu_buffers", self.offload_block_cpu_buffers)
            self.offload_phase_cpu_buffers = None

    def non_block_weights_to_cuda(self):
        return None

    def non_block_weights_to_cpu(self):
        return None


class Cosmos3TransformerLayerWeights(WeightModule):
    def __init__(
        self,
        layer_idx,
        mm_type,
        config,
        rms_norm_type,
        attn_rms_norm_type,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        lora_path=None,
    ):
        super().__init__()
        eps = config.get("rms_norm_eps", 1e-6)
        prefix = f"layers.{layer_idx}"
        self.add_module(
            "self_attn",
            Cosmos3PackedMoTAttentionWeights(
                prefix,
                mm_type,
                config,
                eps,
                attn_rms_norm_type,
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=lazy_load,
                lazy_load_file=lazy_load_file,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "mlp",
            Cosmos3MLPWeights(f"{prefix}.mlp", mm_type, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, lora_path),
        )
        self.add_module(
            "mlp_moe_gen",
            Cosmos3MLPWeights(f"{prefix}.mlp_moe_gen", mm_type, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, lora_path),
        )
        self.add_module(
            "input_layernorm",
            RMS_WEIGHT_REGISTER[rms_norm_type](
                f"{prefix}.input_layernorm.weight",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                eps=eps,
            ),
        )
        self.add_module(
            "input_layernorm_moe_gen",
            RMS_WEIGHT_REGISTER[rms_norm_type](
                f"{prefix}.input_layernorm_moe_gen.weight",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                eps=eps,
            ),
        )
        self.add_module(
            "post_attention_layernorm",
            RMS_WEIGHT_REGISTER[rms_norm_type](
                f"{prefix}.post_attention_layernorm.weight",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                eps=eps,
            ),
        )
        self.add_module(
            "post_attention_layernorm_moe_gen",
            RMS_WEIGHT_REGISTER[rms_norm_type](
                f"{prefix}.post_attention_layernorm_moe_gen.weight",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                eps=eps,
            ),
        )


class Cosmos3PackedMoTAttentionWeights(WeightModule):
    def __init__(
        self,
        prefix,
        mm_type,
        config,
        eps,
        attn_rms_norm_type,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        lora_path=None,
    ):
        super().__init__()
        self.add_module(
            "rope",
            build_rope_weight(config, layout="split_half", default="cosmos3_rope", compute_dtype=torch.float32),
        )
        lora_prefix = "layers"
        attn_type = config.get("self_attn_type", "torch_sdpa")
        causal_attn_type = config.get("causal_self_attn_type", "torch_sdpa")
        self.add_module(
            "to_q",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attn.to_q.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "to_k",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attn.to_k.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "to_v",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attn.to_v.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "to_out",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attn.to_out.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "norm_q",
            RMS_WEIGHT_REGISTER[attn_rms_norm_type](
                f"{prefix}.self_attn.norm_q.weight",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                eps=eps,
            ),
        )
        self.add_module(
            "norm_k",
            RMS_WEIGHT_REGISTER[attn_rms_norm_type](
                f"{prefix}.self_attn.norm_k.weight",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                eps=eps,
            ),
        )

        self.add_module(
            "add_q_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attn.add_q_proj.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "add_k_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attn.add_k_proj.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "add_v_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attn.add_v_proj.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "to_add_out",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attn.to_add_out.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "norm_added_q",
            RMS_WEIGHT_REGISTER[attn_rms_norm_type](
                f"{prefix}.self_attn.norm_added_q.weight",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                eps=eps,
            ),
        )
        self.add_module(
            "norm_added_k",
            RMS_WEIGHT_REGISTER[attn_rms_norm_type](
                f"{prefix}.self_attn.norm_added_k.weight",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                eps=eps,
            ),
        )
        self.add_module("self_attn", ATTN_WEIGHT_REGISTER[attn_type]())
        self.add_module("causal_self_attn", ATTN_WEIGHT_REGISTER[causal_attn_type]())


class Cosmos3MLPWeights(WeightModule):
    def __init__(self, prefix, mm_type, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None, lora_path=None):
        super().__init__()
        lora_prefix = "layers"
        self.add_module(
            "gate_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.gate_proj.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "up_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.up_proj.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "down_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.down_proj.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=lora_prefix,
                lora_path=lora_path,
            ),
        )
