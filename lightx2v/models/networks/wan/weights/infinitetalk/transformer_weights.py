from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.wan.weights.transformer_weights import WanCrossAttention, WanFFN, WanSelfAttention
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, TENSOR_REGISTER


class WanInfiniteTalkTransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        if config.get("cpu_offload", False):
            raise NotImplementedError("InfiniteTalk currently supports cpu_offload=false in LightX2V.")
        if config.get("dit_quantized", False):
            raise NotImplementedError("InfiniteTalk adapter loading currently supports original Wan DiT weights.")

        self.blocks_num = config["num_layers"]
        self.task = config["task"]
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.lazy_load = config.get("lazy_load", False)
        self.blocks = WeightModuleList(
            [
                WanInfiniteTalkTransformerAttentionBlock(
                    block_index=i,
                    task=self.task,
                    mm_type=self.mm_type,
                    config=self.config,
                    block_prefix="blocks",
                    lazy_load=self.lazy_load,
                    lazy_load_path=lazy_load_path,
                    lora_path=lora_path,
                )
                for i in range(self.blocks_num)
            ]
        )
        self.add_module("blocks", self.blocks)

        self.register_parameter("norm", LN_WEIGHT_REGISTER["torch"]())
        self.add_module(
            "head",
            MM_WEIGHT_REGISTER["Default"](
                "head.head.weight",
                "head.head.bias",
                lora_prefix="diffusion_model.head",
            ),
        )
        self.register_parameter("head_modulation", TENSOR_REGISTER["Default"]("head.modulation"))

    def non_block_weights_to_cuda(self):
        self.norm.to_cuda()
        self.head.to_cuda()
        self.head_modulation.to_cuda()

    def non_block_weights_to_cpu(self):
        self.norm.to_cpu()
        self.head.to_cpu()
        self.head_modulation.to_cpu()


class WanInfiniteTalkTransformerAttentionBlock(WeightModule):
    def __init__(
        self,
        block_index,
        task,
        mm_type,
        config,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        block_prefix="blocks",
        lazy_load=False,
        lazy_load_path=None,
        lora_path=None,
    ):
        super().__init__()
        self.compute_phases = WeightModuleList(
            [
                WanSelfAttention(
                    block_index,
                    block_prefix,
                    task,
                    mm_type,
                    config,
                    create_cuda_buffer,
                    create_cpu_buffer,
                    lazy_load,
                    lazy_load_path,
                    lora_path,
                ),
                WanCrossAttention(
                    block_index,
                    block_prefix,
                    task,
                    mm_type,
                    config,
                    create_cuda_buffer,
                    create_cpu_buffer,
                    lazy_load,
                    lazy_load_path,
                    lora_path,
                ),
                WanInfiniteTalkAudioCrossAttention(
                    block_index,
                    block_prefix,
                    config,
                    create_cuda_buffer,
                    create_cpu_buffer,
                    lazy_load,
                    lazy_load_path,
                ),
                WanFFN(
                    block_index,
                    block_prefix,
                    task,
                    mm_type,
                    config,
                    create_cuda_buffer,
                    create_cpu_buffer,
                    lazy_load,
                    lazy_load_path,
                    lora_path,
                ),
            ]
        )
        self.add_module("compute_phases", self.compute_phases)


class WanInfiniteTalkAudioCrossAttention(WeightModule):
    def __init__(
        self,
        block_index,
        block_prefix,
        config,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
    ):
        super().__init__()
        self.block_index = block_index
        self.config = config
        self.add_module(
            "norm_x",
            LN_WEIGHT_REGISTER["torch"](
                f"{block_prefix}.{block_index}.norm_x.weight",
                f"{block_prefix}.{block_index}.norm_x.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
            ),
        )
        self.add_module(
            "q_linear",
            MM_WEIGHT_REGISTER["Default"](
                f"{block_prefix}.{block_index}.audio_cross_attn.q_linear.weight",
                f"{block_prefix}.{block_index}.audio_cross_attn.q_linear.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
            ),
        )
        self.add_module(
            "kv_linear",
            MM_WEIGHT_REGISTER["Default"](
                f"{block_prefix}.{block_index}.audio_cross_attn.kv_linear.weight",
                f"{block_prefix}.{block_index}.audio_cross_attn.kv_linear.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
            ),
        )
        self.add_module(
            "proj",
            MM_WEIGHT_REGISTER["Default"](
                f"{block_prefix}.{block_index}.audio_cross_attn.proj.weight",
                f"{block_prefix}.{block_index}.audio_cross_attn.proj.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
            ),
        )
        self.add_module("audio_attn", ATTN_WEIGHT_REGISTER[config.get("audio_cross_attn_type", config.get("cross_attn_1_type", "flash_attn3"))]())
