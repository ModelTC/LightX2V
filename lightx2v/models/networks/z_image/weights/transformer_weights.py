from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
)


class ZImageTransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        self.blocks_num = config["n_layers"]
        self.task = config["task"]
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        if self.mm_type != "Default":
            assert config.get("dit_quantized") is True
        self.lazy_load = self.config.get("lazy_load", False)
        self.n_refiner_layers = config.get("n_refiner_layers", 0)
        self.register_offload_buffers(config, lazy_load_path, lora_path)
        self.add_module(
            "blocks",
            WeightModuleList(
                ZImageTransformerBlock(i, self.task, self.mm_type, self.config, False, False, "layers", lazy_load=self.lazy_load, lazy_load_path=lazy_load_path) for i in range(self.blocks_num)
            ),
        )

        self.add_module(
            "noise_refiner",
            WeightModuleList(
                ZImageTransformerBlock(
                    i,
                    self.task,
                    self.mm_type,
                    self.config,
                    False,
                    False,
                    "noise_refiner",
                    has_modulation=True,
                )
                for i in range(self.n_refiner_layers)
            ),
        )

        self.add_module(
            "context_refiner",
            WeightModuleList(
                ZImageTransformerBlock(
                    i,
                    self.task,
                    self.mm_type,
                    self.config,
                    False,
                    False,
                    "context_refiner",
                    has_modulation=False,
                )
                for i in range(self.n_refiner_layers)
            ),
        )

    def register_offload_buffers(self, config, lazy_load_path, lora_path):
        if config["cpu_offload"]:
            if config["offload_granularity"] == "block":
                self.offload_blocks_num = 2
                self.offload_block_cuda_buffers = WeightModuleList(
                    [
                        ZImageTransformerBlock(
                            i,
                            self.task,
                            self.mm_type,
                            self.config,
                            True,
                            False,
                            "layers",
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
                            ZImageTransformerBlock(
                                i,
                                self.task,
                                self.mm_type,
                                self.config,
                                False,
                                True,
                                "layers",
                                lazy_load=self.lazy_load,
                                lazy_load_path=lazy_load_path,
                                lora_path=lora_path,
                            )
                            for i in range(self.offload_blocks_num)
                        ]
                    )
                    self.add_module("offload_block_cpu_buffers", self.offload_block_cpu_buffers)
                    self.offload_phase_cpu_buffers = None

    def non_block_weights_to_cuda(self):
        self.noise_refiner.to_cuda()
        self.context_refiner.to_cuda()

    def non_block_weights_to_cpu(self):
        self.noise_refiner.to_cpu()
        self.context_refiner.to_cpu()


class ZImageTransformerBlock(WeightModule):
    """
    Z-Image single stream transformer block.
    Based on ZImageTransformerBlock with attention_norm1, attention_norm2, ffn_norm1, ffn_norm2.
    """

    def __init__(
        self,
        block_idx,
        task,
        mm_type,
        config,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        block_prefix="layers",
        has_modulation=True,
        lazy_load=False,
        lazy_load_path=None,
        lora_path=None,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.has_modulation = has_modulation
        self.ln_type = config.get("ln_type", "Triton")
        self.rms_norm_type = config.get("rms_norm_type", "sgl-kernel")

        self.lazy_load = lazy_load
        if self.lazy_load:
            self.lazy_load_file = lazy_load_path
        else:
            self.lazy_load_file = None

        self.compute_phases = WeightModuleList(
            [
                (
                    ZImageAdaLNModulation(
                        block_idx=block_idx,
                        block_prefix=block_prefix,
                        task=task,
                        mm_type=mm_type,
                        config=config,
                        create_cuda_buffer=create_cuda_buffer,
                        create_cpu_buffer=create_cpu_buffer,
                        lazy_load=self.lazy_load,
                        lazy_load_file=self.lazy_load_file,
                    )
                    if self.has_modulation
                    else WeightModule()
                ),
                ZImageAttention(
                    block_idx=block_idx,
                    block_prefix=block_prefix,
                    task=task,
                    mm_type=mm_type,
                    config=config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                ),
                ZImageFFN(
                    block_idx=block_idx,
                    block_prefix=block_prefix,
                    task=task,
                    mm_type=mm_type,
                    config=config,
                    create_cuda_buffer=create_cuda_buffer,
                    create_cpu_buffer=create_cpu_buffer,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                ),
            ]
        )
        self.add_module("compute_phases", self.compute_phases)


class ZImageAdaLNModulation(WeightModule):
    def __init__(
        self,
        block_idx,
        block_prefix,
        task,
        mm_type,
        config,
        create_cuda_buffer,
        create_cpu_buffer,
        lazy_load,
        lazy_load_file,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.add_module(
            "adaLN_modulation",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{block_idx}.adaLN_modulation.0.weight",
                f"{block_prefix}.{block_idx}.adaLN_modulation.0.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
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


class ZImageAttention(WeightModule):
    """
    Single stream attention for Z-Image.
    Based on ZSingleStreamAttnProcessor.
    """

    def __init__(
        self,
        block_idx,
        block_prefix,
        task,
        mm_type,
        config,
        create_cuda_buffer,
        create_cpu_buffer,
        lazy_load,
        lazy_load_file,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.attn_type = config.get("attn_type", "flash_attn3")
        self.rms_norm_type = config.get("rms_norm_type", "one-pass")

        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        # Attention normalization layers
        self.add_module(
            "attention_norm1",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{block_idx}.attention_norm1.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
            ),
        )
        self.add_module(
            "attention_norm2",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{block_idx}.attention_norm2.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
            ),
        )

        # QK normalization (applied in processor)
        self.add_module(
            "norm_q",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{block_idx}.attention.norm_q.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
            ),
        )
        self.add_module(
            "norm_k",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{block_idx}.attention.norm_k.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
            ),
        )

        # QKV projections
        # Note: Z-Image attention layers don't have bias
        self.add_module(
            "to_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{block_idx}.attention.to_q.weight",
                None,  # No bias in Z-Image
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "to_k",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{block_idx}.attention.to_k.weight",
                None,  # No bias in Z-Image
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "to_v",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{block_idx}.attention.to_v.weight",
                None,  # No bias in Z-Image
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        # Output projection
        self.add_module(
            "to_out",
            WeightModuleList(
                [
                    MM_WEIGHT_REGISTER[self.mm_type](
                        f"{block_prefix}.{block_idx}.attention.to_out.0.weight",
                        None,  # No bias in Z-Image
                        create_cuda_buffer,
                        create_cpu_buffer,
                        self.lazy_load,
                        self.lazy_load_file,
                    ),
                ]
            ),
        )

        # Attention computation
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


class ZImageFFN(WeightModule):
    """
    Feed forward network for Z-Image.
    Based on FeedForward with w1, w2, w3 and SiLU gating.
    """

    def __init__(
        self,
        block_idx,
        block_prefix,
        task,
        mm_type,
        config,
        create_cuda_buffer,
        create_cpu_buffer,
        lazy_load,
        lazy_load_file,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.rms_norm_type = config.get("rms_norm_type", "one-pass")
        # w1, w2, w3 for SiLU gating
        # Note: Z-Image feed_forward layers don't have bias
        self.add_module(
            "w1",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{block_idx}.feed_forward.w1.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "w2",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{block_idx}.feed_forward.w2.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "w3",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{block_idx}.feed_forward.w3.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        self.add_module(
            "ffn_norm1",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{block_idx}.ffn_norm1.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
            ),
        )
        self.add_module(
            "ffn_norm2",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{block_idx}.ffn_norm2.weight",
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
