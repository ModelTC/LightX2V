from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class ErnieImageTransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        self.config = config
        self.blocks_num = config["num_layers"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        if self.mm_type != "Default":
            assert config.get("dit_quantized") is True
        self.lazy_load = config.get("lazy_load", False)
        blocks = WeightModuleList(
            ErnieImageTransformerBlockWeights(
                i,
                self.mm_type,
                config,
                create_cuda_buffer=False,
                create_cpu_buffer=False,
                lazy_load=self.lazy_load,
                lazy_load_path=lazy_load_path,
                lora_path=lora_path,
            )
            for i in range(self.blocks_num)
        )
        self.add_module("blocks", blocks)


class ErnieImageTransformerBlockWeights(WeightModule):
    def __init__(
        self,
        block_index,
        mm_type,
        config,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_path=None,
        lora_path=None,
    ):
        super().__init__()
        prefix = f"layers.{block_index}"
        eps = config.get("eps", 1e-6)
        self.add_module(
            "adaLN_sa_ln",
            RMS_WEIGHT_REGISTER["torch"](
                f"{prefix}.adaLN_sa_ln.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=lazy_load,
                lazy_load_file=lazy_load_path,
                eps=eps,
            ),
        )
        self.add_module(
            "to_q",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attention.to_q.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_path,
                lora_prefix="layers",
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "to_k",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attention.to_k.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_path,
                lora_prefix="layers",
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "to_v",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attention.to_v.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_path,
                lora_prefix="layers",
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "norm_q",
            RMS_WEIGHT_REGISTER["torch"](
                f"{prefix}.self_attention.norm_q.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=lazy_load,
                lazy_load_file=lazy_load_path,
                eps=eps,
            ),
        )
        self.add_module(
            "norm_k",
            RMS_WEIGHT_REGISTER["torch"](
                f"{prefix}.self_attention.norm_k.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=lazy_load,
                lazy_load_file=lazy_load_path,
                eps=eps,
            ),
        )
        self.add_module(
            "to_out",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.self_attention.to_out.0.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_path,
                lora_prefix="layers",
                lora_path=lora_path,
            ),
        )
        self.add_module("attn", ATTN_WEIGHT_REGISTER[config.get("attn_type", "torch_sdpa")]())
        self.add_module(
            "adaLN_mlp_ln",
            RMS_WEIGHT_REGISTER["torch"](
                f"{prefix}.adaLN_mlp_ln.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=lazy_load,
                lazy_load_file=lazy_load_path,
                eps=eps,
            ),
        )
        self.add_module(
            "gate_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.mlp.gate_proj.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_path,
                lora_prefix="layers",
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "up_proj",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.mlp.up_proj.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_path,
                lora_prefix="layers",
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "linear_fc2",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.mlp.linear_fc2.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_path,
                lora_prefix="layers",
                lora_path=lora_path,
            ),
        )
