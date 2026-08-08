import torch
import torch.distributed as dist

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
    ROPE_REGISTER,
)


def _resolve_resident_block_indices(value, num_blocks, policy="interleaved"):
    if value is None:
        value = 0
    if isinstance(value, str):
        if value.lower() != "all":
            raise ValueError(f"offload_resident_blocks must be an integer or 'all', got {value!r}")
        count = num_blocks
    elif isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"offload_resident_blocks must be an integer or 'all', got {value!r}")
    else:
        count = value

    if not 0 <= count <= num_blocks:
        raise ValueError(f"offload_resident_blocks must be between 0 and {num_blocks}, got {count}")
    if count == 0:
        return frozenset()
    if count == num_blocks:
        return frozenset(range(num_blocks))
    if policy == "prefix":
        return frozenset(range(count))
    if policy == "interleaved":
        return frozenset((idx * num_blocks) // count for idx in range(count))
    raise ValueError(f"offload_resident_policy must be 'prefix' or 'interleaved', got {policy!r}")


def release_weight_module_device_tensors(module):
    """Drop immutable device weights and retain their pinned CPU masters."""
    for child in getattr(module, "_modules", {}).values():
        if child is not None:
            release_weight_module_device_tensors(child)

    for _, attr_name, _ in getattr(module, "base_attrs", ()):
        value = getattr(module, attr_name, None)
        pin_value = getattr(module, f"pin_{attr_name}", None)
        if pin_value is not None:
            setattr(module, attr_name, None)
        elif isinstance(value, torch.Tensor) and value.device.type != "cpu":
            setattr(module, attr_name, value.to("cpu"))


class QwenImageTransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        if config.get("use_compile") and config.get("rms_norm_type", "one-pass") == "one-pass":
            raise ValueError("Qwen Image torch.compile does not support one-pass RMSNorm.")
        self.blocks_num = config["num_layers"]
        self.task = config["task"]
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        if self.mm_type != "Default":
            assert config.get("dit_quantized") is True
        self.lazy_load = self.config.get("lazy_load", False)
        self._configure_resident_blocks(config)
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
                if len(self.resident_block_indices) < self.blocks_num:
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

    def _configure_resident_blocks(self, config):
        block_offload_enabled = config.get("cpu_offload", False) and config.get("offload_granularity", "block") == "block"
        resident_setting = config.get("offload_resident_blocks", 0) if block_offload_enabled else 0
        if block_offload_enabled and dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            resident_setting = config.get("offload_resident_blocks_rank0", resident_setting)
        if resident_setting not in (None, 0):
            if config.get("dit_quantized", False):
                raise NotImplementedError("Qwen-Image resident block offload currently supports unquantized weights only")
            if config.get("lora_configs"):
                raise NotImplementedError("Qwen-Image resident block offload currently does not support LoRA weights")
        self.resident_block_indices = _resolve_resident_block_indices(
            resident_setting,
            self.blocks_num,
            config.get("offload_resident_policy", "interleaved"),
        )

    def resident_blocks_to_cuda(self, non_blocking=True):
        for block_idx in sorted(self.resident_block_indices):
            self.blocks[block_idx].to_cuda(non_blocking=non_blocking)

    def release_resident_blocks(self):
        for block_idx in sorted(self.resident_block_indices):
            release_weight_module_device_tensors(self.blocks[block_idx])


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
            "rope",
            ROPE_REGISTER[config.get("rope_type", "flashinfer_rope")](layout="interleaved", compute_dtype=torch.float32),
        )

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
            "rope",
            ROPE_REGISTER[config.get("rope_type", "flashinfer_rope")](layout="interleaved", compute_dtype=torch.float32),
        )

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
