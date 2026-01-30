from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import (
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
)


class Qwen2TransformerWeights(WeightModule):
    def __init__(self, config, llm_config, lazy_load_path=None):
        super().__init__()
        self.config = config
        self.blocks_num = llm_config["num_hidden_layers"]
        self.task = config["task"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        if self.mm_type != "Default":
            assert config.get("dit_quantized") is True
        self.lazy_load = self.config.get("lazy_load", False)
        blocks = WeightModuleList(
            Qwen2MoTDecoderLayer(
                i,
                self.task,
                self.mm_type,
                self.config,
            ) for i in range(self.blocks_num)
        )
        self.add_module("blocks", blocks)


class Qwen2MoTDecoderLayer(WeightModule):
    def __init__(
        self,
        block_index,
        task,
        mm_type,
        config,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_path=None,
    ):
        super().__init__()
        self.config = config
        # input_layernorm
        self.add_module(
            "input_layernorm",
            RMS_WEIGHT_REGISTER["fp32_variance"](f"language_model.model.layers.{block_index}.input_layernorm.weight"),
        )
        self.add_module(
            "input_layernorm_moe_gen",
            RMS_WEIGHT_REGISTER["fp32_variance"](f"language_model.model.layers.{block_index}.input_layernorm_moe_gen.weight"),
        )
        # mlp
        mlp = Qwen2MLP(
            block_index=block_index,
            task=task,
            mm_type=mm_type,
            config=config,
            subname="mlp",
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=create_cpu_buffer,
            lazy_load=lazy_load,
            lazy_load_path=lazy_load_path
        )
        self.add_module("mlp", mlp)
        mlp_moe_gen = Qwen2MLP(
            block_index=block_index,
            task=task,
            mm_type=mm_type,
            config=config,
            subname="mlp_moe_gen",
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=create_cpu_buffer,
            lazy_load=lazy_load,
            lazy_load_path=lazy_load_path
        )
        self.add_module("mlp_moe_gen", mlp_moe_gen)
        # post_attention_layernorm
        self.add_module(
            "post_attention_layernorm",
            RMS_WEIGHT_REGISTER["fp32_variance"](f"language_model.model.layers.{block_index}.post_attention_layernorm.weight"),
        )
        self.add_module(
            "post_attention_layernorm_moe_gen",
            RMS_WEIGHT_REGISTER["fp32_variance"](f"language_model.model.layers.{block_index}.post_attention_layernorm_moe_gen.weight"),
        )
        # self attn
        attn = PackedAttentionMoT(
            block_index=block_index,
            task=task,
            mm_type=mm_type,
            config=config   
        )
        self.add_module("self_attn", attn)


class Qwen2MLP(WeightModule):
    def __init__(
        self,
        block_index,
        task,
        mm_type,
        config,
        subname="mlp",
        create_cuda_buffer=False,
        create_cpu_buffer=False,  
        lazy_load=False,
        lazy_load_path=None,
    ):
        super().__init__()
        self.config = config
        self.add_module(
            "gate_proj",
            MM_WEIGHT_REGISTER[mm_type](f"language_model.model.layers.{block_index}.{subname}.gate_proj.weight"),
        )
        self.add_module(
            "up_proj",
            MM_WEIGHT_REGISTER[mm_type](f"language_model.model.layers.{block_index}.{subname}.up_proj.weight"),
        )
        self.add_module(
            "down_proj",
            MM_WEIGHT_REGISTER[mm_type](f"language_model.model.layers.{block_index}.{subname}.down_proj.weight"),
        )
        

class PackedAttentionMoT(WeightModule):
    def __init__(
        self,
        block_index,
        task,
        mm_type,
        config,
        create_cuda_buffer=False,
        create_cpu_buffer=False,  
        lazy_load=False,
        lazy_load_path=None,
    ):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        # q
        self.add_module(
            "q_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"language_model.model.layers.{block_index}.self_attn.q_proj.weight",
                f"language_model.model.layers.{block_index}.self_attn.q_proj.bias",
            ),
        )
        self.add_module(
            "q_norm",
            RMS_WEIGHT_REGISTER["fp32_variance"](f"language_model.model.layers.{block_index}.self_attn.q_norm.weight"),
        )
        self.add_module(
            "q_norm_moe_gen",
            RMS_WEIGHT_REGISTER["fp32_variance"](f"language_model.model.layers.{block_index}.self_attn.q_norm_moe_gen.weight"),
        )
        self.add_module(
            "q_proj_moe_gen",
            MM_WEIGHT_REGISTER[mm_type](
                f"language_model.model.layers.{block_index}.self_attn.q_proj_moe_gen.weight",
                f"language_model.model.layers.{block_index}.self_attn.q_proj_moe_gen.bias",
            ),
        )
        # k
        self.add_module(
            "k_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"language_model.model.layers.{block_index}.self_attn.k_proj.weight",
                f"language_model.model.layers.{block_index}.self_attn.k_proj.bias",
            ),
        )
        self.add_module(
            "k_norm",
            RMS_WEIGHT_REGISTER["fp32_variance"](f"language_model.model.layers.{block_index}.self_attn.k_norm.weight"),
        )
        self.add_module(
            "k_norm_moe_gen",
            RMS_WEIGHT_REGISTER["fp32_variance"](f"language_model.model.layers.{block_index}.self_attn.k_norm_moe_gen.weight"),
        )
        self.add_module(
            "k_proj_moe_gen",
            MM_WEIGHT_REGISTER[mm_type](
                f"language_model.model.layers.{block_index}.self_attn.k_proj_moe_gen.weight",
                f"language_model.model.layers.{block_index}.self_attn.k_proj_moe_gen.bias",
            ),
        )
        # v
        self.add_module(
            "v_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"language_model.model.layers.{block_index}.self_attn.v_proj.weight",
                f"language_model.model.layers.{block_index}.self_attn.v_proj.bias",
            ),
        )
        self.add_module(
            "v_proj_moe_gen",
            MM_WEIGHT_REGISTER[mm_type](
                f"language_model.model.layers.{block_index}.self_attn.v_proj_moe_gen.weight",
                f"language_model.model.layers.{block_index}.self_attn.v_proj_moe_gen.bias",
            ),
        )
        # o
        self.add_module(
            "o_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"language_model.model.layers.{block_index}.self_attn.o_proj.weight"
            )
        )
        self.add_module(
            "o_proj_moe_gen",
            MM_WEIGHT_REGISTER[mm_type](
                f"language_model.model.layers.{block_index}.self_attn.o_proj_moe_gen.weight"
            )
        )
        
        