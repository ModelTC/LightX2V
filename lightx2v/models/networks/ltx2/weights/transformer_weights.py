import torch.distributed as dist

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
        self.register_offload_buffers(config, lazy_load_path, lora_path)
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

    def register_offload_buffers(self, config, lazy_load_path, lora_path):
        if config["cpu_offload"]:
            if config["offload_granularity"] == "block":
                self.offload_blocks_num = 2
                self.offload_block_cuda_buffers = WeightModuleList(
                    [
                        LTX2TransformerBlock(
                            block_index=i,
                            task=self.task,
                            mm_type=self.mm_type if i not in self.skip_fp8_block_index else "Default",
                            config=self.config,
                            create_cuda_buffer=True,
                            create_cpu_buffer=False,
                            block_prefix="transformer_blocks",
                            lazy_load=self.lazy_load,
                            lazy_load_path=lazy_load_path,
                        )
                        for i in range(self.offload_blocks_num)
                    ]
                )
                self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)
                self.offload_phase_cuda_buffers = None


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

        # Check if tensor parallel is enabled
        use_tp = config.get("tensor_parallel", False)
        if use_tp:
            tp_group = config.get("device_mesh").get_group(mesh_dim="tensor_p")
            tp_rank = dist.get_rank(tp_group)
            tp_size = dist.get_world_size(tp_group)
        else:
            tp_group = None
            tp_rank = 0
            tp_size = 1

        # Create attention and FFN modules based on tensor parallel config
        if use_tp:
            AttentionClass = LTX2AttentionTP
            FFNClass = LTX2FFNTP
            tp_kwargs = {
                "tp_group": tp_group,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            }
        else:
            AttentionClass = LTX2Attention
            FFNClass = LTX2FFN
            tp_kwargs = {}

        self.compute_phases = WeightModuleList(
            [
                AttentionClass(
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
                    **tp_kwargs,
                ),
                AttentionClass(
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
                    **tp_kwargs,
                ),
                AttentionClass(
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
                    **tp_kwargs,
                ),
                AttentionClass(
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
                    **tp_kwargs,
                ),
                AttentionClass(
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
                    **tp_kwargs,
                ),
                AttentionClass(
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
                    **tp_kwargs,
                ),
                FFNClass(
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
                    **tp_kwargs,
                ),
                FFNClass(
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
                    **tp_kwargs,
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
        self.attn_rms_norm_type = self.config.get("rms_norm_type", "sgl-kernel")

        block_lora_prefix = "model.diffusion_model.blocks"
        model_prefix = "model.diffusion_model"

        self.add_module("attn_func", ATTN_WEIGHT_REGISTER[self.config["attn_type"]]())

        # Add parallel attention module for sequence parallelism
        if self.config.get("seq_parallel", False):
            self.add_module(
                "attn_func_parallel",
                ATTN_WEIGHT_REGISTER[self.config.get("parallel", {}).get("seq_p_attn_type", "ulysses")](),
            )

        self.add_module(
            f"q_norm",
            RMS_WEIGHT_REGISTER[self.attn_rms_norm_type](
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
            RMS_WEIGHT_REGISTER[self.attn_rms_norm_type](
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
        block_lora_prefix = "model.diffusion_model.blocks"
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


class LTX2AttentionTP(WeightModule):
    """
    Tensor Parallel version of LTX2Attention.

    QKV projections are split column-wise, output projection is split row-wise.
    """

    def __init__(
        self,
        block_index,
        attn_prefix,
        block_prefix,
        task,
        mm_type,
        config,
        tp_group=None,
        tp_rank=0,
        tp_size=1,
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
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.attn_rms_norm_type = self.config.get("rms_norm_type", "sgl-kernel")

        block_lora_prefix = "model.diffusion_model.blocks"
        model_prefix = "model.diffusion_model"

        self.add_module("attn_func", ATTN_WEIGHT_REGISTER[self.config["attn_type"]]())

        # Use TP version of norm if tensor parallel is enabled
        # Note: In TP, QKV outputs are split, so norm weights must also be split
        norm_class = RMS_WEIGHT_REGISTER["TensorParallel"]
        norm_kwargs = {
            "tp_group": tp_group,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
        }

        self.add_module(
            f"q_norm",
            norm_class(
                weight_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.q_norm.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
                **norm_kwargs,
            ),
        )
        self.add_module(
            f"k_norm",
            norm_class(
                weight_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.k_norm.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
                **norm_kwargs,
            ),
        )
        # QKV projections: column split
        self.add_module(
            f"to_q",
            MM_WEIGHT_REGISTER["TensorParallel"](
                weight_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_q.weight",
                bias_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_q.bias",
                mm_type=mm_type,
                tp_group=tp_group,
                tp_rank=tp_rank,
                tp_size=tp_size,
                split_dim="col",
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
            MM_WEIGHT_REGISTER["TensorParallel"](
                weight_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_k.weight",
                bias_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_k.bias",
                mm_type=mm_type,
                tp_group=tp_group,
                tp_rank=tp_rank,
                tp_size=tp_size,
                split_dim="col",
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
            MM_WEIGHT_REGISTER["TensorParallel"](
                weight_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_v.weight",
                bias_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_v.bias",
                mm_type=mm_type,
                tp_group=tp_group,
                tp_rank=tp_rank,
                tp_size=tp_size,
                split_dim="col",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
            ),
        )
        # Output projection: row split (needs all-reduce)
        self.add_module(
            f"to_out",
            MM_WEIGHT_REGISTER["TensorParallel"](
                weight_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_out.0.weight",
                bias_name=f"{model_prefix}.{block_prefix}.{block_index}.{attn_prefix}.to_out.0.bias",
                mm_type=mm_type,
                tp_group=tp_group,
                tp_rank=tp_rank,
                tp_size=tp_size,
                split_dim="row",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
            ),
        )


class LTX2FFNTP(WeightModule):
    """
    Tensor Parallel version of LTX2FFN.

    First layer (net_0_proj) is split column-wise, second layer (net_2) is split row-wise.
    """

    def __init__(
        self,
        block_index,
        block_prefix,
        ffn_prefix,
        task,
        mm_type,
        config,
        tp_group=None,
        tp_rank=0,
        tp_size=1,
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
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        block_lora_prefix = "model.diffusion_model.blocks"
        model_prefix = "model.diffusion_model"

        # First layer: column split
        self.add_module(
            f"net_0_proj",
            MM_WEIGHT_REGISTER["TensorParallel"](
                f"{model_prefix}.{block_prefix}.{block_index}.{ffn_prefix}.net.0.proj.weight",
                f"{model_prefix}.{block_prefix}.{block_index}.{ffn_prefix}.net.0.proj.bias",
                mm_type=mm_type,
                tp_group=tp_group,
                tp_rank=tp_rank,
                tp_size=tp_size,
                split_dim="col",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
            ),
        )
        # Second layer: row split (needs all-reduce)
        self.add_module(
            f"net_2",
            MM_WEIGHT_REGISTER["TensorParallel"](
                f"{model_prefix}.{block_prefix}.{block_index}.{ffn_prefix}.net.2.weight",
                f"{model_prefix}.{block_prefix}.{block_index}.{ffn_prefix}.net.2.bias",
                mm_type=mm_type,
                tp_group=tp_group,
                tp_rank=tp_rank,
                tp_size=tp_size,
                split_dim="row",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
                lora_prefix=block_lora_prefix,
                lora_path=lora_path,
            ),
        )
