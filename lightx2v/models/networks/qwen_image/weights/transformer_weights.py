from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
)


class QwenImageTransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        self.blocks_num = config["num_layers"]
        self.task = config["task"]
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        if self.mm_type != "Default":
            assert config.get("dit_quantized") is True
        self.lazy_load = self.config.get("lazy_load", False)
        blocks = WeightModuleList(
            QwenImageTransformerAttentionBlock(
                i,
                self.task,
                self.mm_type,
                self.config,
                False,
                False,
                "transformer_blocks",
                lazy_load=self.lazy_load,
                lazy_load_path=lazy_load_path,
            )
            for i in range(self.blocks_num)
        )
        self.register_offload_buffers(config, lazy_load_path, lora_path)
        self.add_module("blocks", blocks)

    def register_offload_buffers(self, config, lazy_load_path, lora_path):
        if config["cpu_offload"]:
            if config["offload_granularity"] == "block":
                self.offload_blocks_num = 2
                self.offload_block_cuda_buffers = WeightModuleList(
                    [
                        QwenImageTransformerAttentionBlock(
                            i,
                            self.task,
                            self.mm_type,
                            self.config,
                            True,
                            False,
                            "transformer_blocks",
                            lazy_load=self.lazy_load,
                            lazy_load_path=lazy_load_path,
                        )
                        for i in range(self.offload_blocks_num)
                    ]
                )
                self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)
                self.offload_phase_cuda_buffers = None
                if self.lazy_load:
                    self.offload_blocks_num = 2
                    self.offload_block_cpu_buffers = WeightModuleList(
                        [
                            QwenImageTransformerAttentionBlock(
                                i,
                                self.task,
                                self.mm_type,
                                self.config,
                                False,
                                True,
                                "transformer_blocks",
                                lazy_load=self.lazy_load,
                                lazy_load_path=lazy_load_path,
                                lora_path=lora_path,
                            )
                            for i in range(self.offload_blocks_num)
                        ]
                    )
                    self.add_module("offload_block_cpu_buffers", self.offload_block_cpu_buffers)
                    self.offload_phase_cpu_buffers = None

            elif config["offload_granularity"] == "phase":
                self.offload_phase_cuda_buffers = QwenImageTransformerAttentionBlock(
                    0,
                    self.task,
                    self.mm_type,
                    self.config,
                    True,
                    False,
                    "transformer_blocks",
                    lazy_load=self.lazy_load,
                    lazy_load_path=lazy_load_path,
                ).compute_phases
                self.add_module("offload_phase_cuda_buffers", self.offload_phase_cuda_buffers)
                self.offload_block_cuda_buffers = None
                if self.lazy_load:
                    self.offload_phase_cpu_buffers = WeightModuleList(
                        [
                            QwenImageTransformerAttentionBlock(
                                i, self.task, self.mm_type, self.config, False, True, "transformer_blocks", lazy_load=self.lazy_load, lazy_load_path=lazy_load_path, lora_path=lora_path
                            ).compute_phases
                            for i in range(2)
                        ]
                    )
                    self.add_module("offload_phase_cpu_buffers", self.offload_phase_cpu_buffers)
                    self.offload_block_cpu_buffers = None


class QwenImageTransformerAttentionBlock(WeightModule):
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
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.lazy_load = lazy_load
        if self.lazy_load:
            self.lazy_load_file = lazy_load_path
        else:
            self.lazy_load_file = None

        self.compute_phases = WeightModuleList(
            [
                QwenImageImgAttention(
                    block_index=block_index,
                    block_prefix=block_prefix,
                    task=config["task"],
                    mm_type=mm_type,
                    config=config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                    lora_path=lora_path,
                ),
                QwenImageTxtAttention(
                    block_index=block_index,
                    block_prefix=block_prefix,
                    task=config["task"],
                    mm_type=mm_type,
                    config=config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                    lora_path=lora_path,
                ),
                QwenImageCrossAttention(
                    block_index=block_index,
                    block_prefix=block_prefix,
                    task=config["task"],
                    mm_type=mm_type,
                    config=config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                    lora_path=lora_path,
                ),
                QwenImageFFN(
                    block_index=block_index,
                    block_prefix=block_prefix,
                    task=config["task"],
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


class QwenImageImgAttention(WeightModule):
    def __init__(
        self,
        block_index,
        block_prefix,
        task,
        mm_type,
        config,
        create_cuda_buffer,
        create_cpu_buffer,
        lazy_load,
        lazy_load_file,
        lora_path,
    ):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.heads = config["num_attention_heads"]
        self.rms_norm_type = config.get("rms_norm_type", "one-pass")
        self.layer_norm_type = config.get("layer_norm_type", "Triton")
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        self.add_module(
            "img_mod",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.img_mod.1.weight",
                f"{block_prefix}.{self.block_index}.img_mod.1.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "img_norm1",
            LN_WEIGHT_REGISTER[self.layer_norm_type](
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                eps=1e-6,
            ),
        )

        self.add_module(
            "to_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.to_q.weight",
                f"{block_prefix}.{self.block_index}.attn.to_q.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
        # to_k
        self.add_module(
            "to_k",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.to_k.weight",
                f"{block_prefix}.{self.block_index}.attn.to_k.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
        # to_v
        self.add_module(
            "to_v",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.to_v.weight",
                f"{block_prefix}.{self.block_index}.attn.to_v.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )

        # norm_q
        self.add_module(
            "norm_q",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{block_index}.attn.norm_q.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
            ),
        )
        # norm_k
        self.add_module(
            "norm_k",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{block_index}.attn.norm_k.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
            ),
        )

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)


class QwenImageTxtAttention(WeightModule):
    def __init__(
        self,
        block_index,
        block_prefix,
        task,
        mm_type,
        config,
        create_cuda_buffer,
        create_cpu_buffer,
        lazy_load,
        lazy_load_file,
        lora_path,
    ):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.heads = config["num_attention_heads"]
        self.rms_norm_type = config.get("rms_norm_type", "one-pass")
        self.layer_norm_type = config.get("layer_norm_type", "Triton")
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        self.add_module(
            "txt_mod",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.txt_mod.1.weight",
                f"{block_prefix}.{self.block_index}.txt_mod.1.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "txt_norm1",
            LN_WEIGHT_REGISTER[self.layer_norm_type](
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                eps=1e-6,
            ),
        )
        # add_q_proj
        self.add_module(
            "add_q_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.add_q_proj.weight",
                f"{block_prefix}.{self.block_index}.attn.add_q_proj.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
        # add_k_proj
        self.add_module(
            "add_k_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.add_k_proj.weight",
                f"{block_prefix}.{self.block_index}.attn.add_k_proj.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
        # add_v_proj
        self.add_module(
            "add_v_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.add_v_proj.weight",
                f"{block_prefix}.{self.block_index}.attn.add_v_proj.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )

        # norm_added_q
        self.add_module(
            "norm_added_q",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{block_index}.attn.norm_added_q.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
            ),
        )
        # norm_added_k
        self.add_module(
            "norm_added_k",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{block_index}.attn.norm_added_k.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
            ),
        )

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)


class QwenImageCrossAttention(WeightModule):
    def __init__(
        self,
        block_index,
        block_prefix,
        task,
        mm_type,
        config,
        create_cuda_buffer,
        create_cpu_buffer,
        lazy_load,
        lazy_load_file,
        lora_path,
    ):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.attn_type = config.get("attn_type", "flash_attn3")
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        # to_out
        self.add_module(
            "to_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.to_out.0.weight",
                f"{block_prefix}.{self.block_index}.attn.to_out.0.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
        # to_add_out
        self.add_module(
            "to_add_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.to_add_out.weight",
                f"{block_prefix}.{self.block_index}.attn.to_add_out.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )

        # attn
        self.add_module("calculate", ATTN_WEIGHT_REGISTER[self.attn_type]())

        if self.config["seq_parallel"]:
            self.add_module(
                "calculate_parallel",
                ATTN_WEIGHT_REGISTER[self.config["parallel"].get("seq_p_attn_type", "ulysses")](),
            )

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)


class QwenImageFFN(WeightModule):
    def __init__(self, block_index, block_prefix, task, mm_type, config, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, lora_path):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.layer_norm_type = config.get("layer_norm_type", "Triton")
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        self.add_module(
            "img_norm2",
            LN_WEIGHT_REGISTER[self.layer_norm_type](
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                eps=1e-6,
            ),
        )

        self.add_module(
            "img_mlp_0",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.img_mlp.net.0.proj.weight",
                f"{block_prefix}.{self.block_index}.img_mlp.net.0.proj.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "img_mlp_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.img_mlp.net.2.weight",
                f"{block_prefix}.{self.block_index}.img_mlp.net.2.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )

        self.add_module(
            "txt_norm2",
            LN_WEIGHT_REGISTER[self.layer_norm_type](
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                eps=1e-6,
            ),
        )

        self.add_module(
            "txt_mlp_0",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.txt_mlp.net.0.proj.weight",
                f"{block_prefix}.{self.block_index}.txt_mlp.net.0.proj.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
        self.add_module(
            "txt_mlp_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.txt_mlp.net.2.weight",
                f"{block_prefix}.{self.block_index}.txt_mlp.net.2.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)
