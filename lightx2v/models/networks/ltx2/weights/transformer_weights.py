from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
    TENSOR_REGISTER,
)


class LTX2TransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        self.blocks_num = config["num_layers"]
        self.task = config["task"]
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        if self.mm_type != "Default":
            assert config.get("dit_quantized") is True
        if config.get("do_mm_calib", False):
            self.mm_type = "Calib"
            assert not config["cpu_offload"]
        self.lazy_load = self.config.get("lazy_load", False)
        self.skip_fp8_block_index = self.config.get("skip_fp8_block_index", [])
        self.blocks = WeightModuleList(
            [
                LTX2TransformerBlock(
                    block_index=i,
                    task=self.task,
                    mm_type=self.mm_type if i not in self.skip_fp8_block_index else "Default",
                    config=self.config,
                    create_cuda_buffer=False,
                    create_cpu_buffer=False,
                    block_prefix="transformer_blocks",
                    lazy_load=self.lazy_load,
                    lazy_load_path=lazy_load_path,
                )
                for i in range(self.blocks_num)
            ]
        )
        self.add_module("blocks", self.blocks)


class LTX2TransformerBlock(WeightModule):
    def __init__(
        self,
        block_index,
        task,
        mm_type,
        config,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        block_prefix="transformer_blocks",
        lazy_load=False,
        lazy_load_path=None,
        lora_path=None,
    ):
        super().__init__()
        self.block_index = block_index
        self.config = config
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_path
        block_prefix = "transformer_blocks"
        model_prefix = "model.diffusion_model"

        # Video scale-shift table
        self.scale_shift_table = TENSOR_REGISTER["Default"](
            tensor_name=f"{model_prefix}.{block_prefix}.{self.block_index}.scale_shift_table",
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=create_cpu_buffer,
            lazy_load=self.lazy_load,
            lazy_load_file=self.lazy_load_file,
        )
        self.add_module(
            "scale_shift_table",
            self.scale_shift_table,
        )

        # Audio scale-shift table
        self.audio_scale_shift_table = TENSOR_REGISTER["Default"](
            tensor_name=f"{model_prefix}.{block_prefix}.{self.block_index}.audio_scale_shift_table",
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=create_cpu_buffer,
            lazy_load=self.lazy_load,
            lazy_load_file=self.lazy_load_file,
        )
        self.add_module(
            "audio_scale_shift_table",
            self.audio_scale_shift_table,
        )

        # Audio-to-video cross-attention audio scale-shift table
        self.scale_shift_table_a2v_ca_audio = TENSOR_REGISTER["Default"](
            tensor_name=f"{model_prefix}.{block_prefix}.{self.block_index}.scale_shift_table_a2v_ca_audio",
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=create_cpu_buffer,
            lazy_load=self.lazy_load,
            lazy_load_file=self.lazy_load_file,
        )
        self.add_module(
            "scale_shift_table_a2v_ca_audio",
            self.scale_shift_table_a2v_ca_audio,
        )

        # Audio-to-video cross-attention video scale-shift table
        self.scale_shift_table_a2v_ca_video = TENSOR_REGISTER["Default"](
            tensor_name=f"{model_prefix}.{block_prefix}.{self.block_index}.scale_shift_table_a2v_ca_video",
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=create_cpu_buffer,
            lazy_load=self.lazy_load,
            lazy_load_file=self.lazy_load_file,
        )
        self.add_module(
            "scale_shift_table_a2v_ca_video",
            self.scale_shift_table_a2v_ca_video,
        )

        self.compute_phases = WeightModuleList(
            [
                LTX2Attention(
                    block_index=block_index,
                    attn_prefix="attn1",
                    block_prefix=block_prefix,
                    task=task,
                    mm_type=mm_type,
                    config=config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                    lora_path=lora_path,
                ),
                LTX2Attention(
                    block_index=block_index,
                    attn_prefix="attn2",
                    block_prefix=block_prefix,
                    task=task,
                    mm_type=mm_type,
                    config=config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                    lora_path=lora_path,
                ),
                LTX2Attention(
                    block_index=block_index,
                    attn_prefix="audio_attn1",
                    block_prefix=block_prefix,
                    task=task,
                    mm_type=mm_type,
                    config=config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                    lora_path=lora_path,
                ),
                LTX2Attention(
                    block_index=block_index,
                    attn_prefix="audio_attn2",
                    block_prefix=block_prefix,
                    task=task,
                    mm_type=mm_type,
                    config=config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                    lora_path=lora_path,
                ),
                LTX2Attention(
                    block_index=block_index,
                    attn_prefix="audio_to_video_attn",
                    block_prefix=block_prefix,
                    task=task,
                    mm_type=mm_type,
                    config=config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                    lora_path=lora_path,
                ),
                LTX2Attention(
                    block_index=block_index,
                    attn_prefix="video_to_audio_attn",
                    block_prefix=block_prefix,
                    task=task,
                    mm_type=mm_type,
                    config=config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                    lora_path=lora_path,
                ),
                LTX2FFN(
                    block_index=block_index,
                    ffn_prefix="ff",
                    block_prefix=block_prefix,
                    task=task,
                    mm_type=mm_type,
                    config=config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                    lora_path=lora_path,
                ),
                LTX2FFN(
                    block_index=block_index,
                    ffn_prefix="audio_ff",
                    block_prefix=block_prefix,
                    task=task,
                    mm_type=mm_type,
                    config=config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                    lora_path=lora_path,
                ),
            ]
        )
        self.add_module("compute_phases", self.compute_phases)

        # Create aliases for easier access
        # The modules are stored with names like "attn1_to_q", "attn1_to_k", etc.
        # These are automatically created by LTX2Attention and LTX2FFN classes


class LTX2Attention(WeightModule):
    def __init__(
        self,
        block_index,
        attn_prefix,
        block_prefix,
        task,
        mm_type,
        config,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        lora_path=None,
    ):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.quant_method = config.get("quant_method", None)

        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.attn_rms_type = self.config.get("rms_type", "sgl-kernel")

        block_lora_prefix = "diffusion_model.blocks"
        model_prefix = "model.diffusion_model"

        self.add_module("attn_func", ATTN_WEIGHT_REGISTER[self.config["attn_type"]]())
        self.add_module(
            f"q_norm",
            RMS_WEIGHT_REGISTER[self.attn_rms_type](
                weight_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.q_norm.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            f"k_norm",
            RMS_WEIGHT_REGISTER[self.attn_rms_type](
                weight_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.k_norm.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            f"to_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                weight_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_q.weight",
                bias_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_q.bias",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            f"to_k",
            MM_WEIGHT_REGISTER[self.mm_type](
                weight_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_k.weight",
                bias_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_k.bias",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            f"to_v",
            MM_WEIGHT_REGISTER[self.mm_type](
                weight_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_v.weight",
                bias_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_v.bias",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            f"to_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                weight_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_out.0.weight",
                bias_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_out.0.bias",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
            ),
        )


class LTX2FFN(WeightModule):
    def __init__(
        self,
        block_index,
        block_prefix,
        ffn_prefix,
        task,
        mm_type,
        config,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        lora_path=None,
    ):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.quant_method = config.get("quant_method", None)
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        block_lora_prefix = "diffusion_model.blocks"
        model_prefix = "model.diffusion_model"

        self.add_module(
            f"net_0_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{model_prefix}.{block_prefix}.{block_index}.{ffn_prefix}.net.0.proj.weight",
                f"{model_prefix}.{block_prefix}.{block_index}.{ffn_prefix}.net.0.proj.bias",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            f"net_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{model_prefix}.{block_prefix}.{block_index}.{ffn_prefix}.net.2.weight",
                f"{model_prefix}.{block_prefix}.{block_index}.{ffn_prefix}.net.2.bias",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
            ),
        )
