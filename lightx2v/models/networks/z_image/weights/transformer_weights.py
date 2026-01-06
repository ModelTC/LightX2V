import os

import torch.nn.functional as F
from safetensors import safe_open

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class ZImageTransformerWeights(WeightModule):
    """
    Z-Image single stream transformer weights.
    Based on ZImageTransformer2DModel structure.
    """

    def __init__(self, config):
        super().__init__()
        self.blocks_num = config["num_layers"]
        self.task = config["task"]
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        if self.mm_type != "Default":
            assert config.get("dit_quantized") is True

        # Main transformer blocks
        blocks = WeightModuleList(ZImageTransformerBlock(i, self.task, self.mm_type, self.config, False, False, "layers") for i in range(self.blocks_num))
        self.add_module("blocks", blocks)

        # Noise refiner (if exists)
        n_refiner_layers = config.get("n_refiner_layers", 0)
        if n_refiner_layers > 0:
            noise_refiner = WeightModuleList(
                ZImageTransformerBlock(
                    i,  # layer_id should be the index i for noise_refiner (not 1000 + i)
                    self.task,
                    self.mm_type,
                    self.config,
                    False,
                    False,
                    "noise_refiner",
                    modulation=True,
                )
                for i in range(n_refiner_layers)
            )
            self.add_module("noise_refiner", noise_refiner)
        else:
            self.noise_refiner = None

        # Context refiner (if exists)
        if n_refiner_layers > 0:
            context_refiner = WeightModuleList(
                ZImageTransformerBlock(
                    i,  # layer_id should be the index i for context_refiner
                    self.task,
                    self.mm_type,
                    self.config,
                    False,
                    False,
                    "context_refiner",
                    modulation=False,
                )
                for i in range(n_refiner_layers)
            )
            self.add_module("context_refiner", context_refiner)
        else:
            self.context_refiner = None

        self.register_offload_buffers(config)

    def register_offload_buffers(self, config):
        if config["cpu_offload"]:
            if config["offload_granularity"] == "block":
                self.offload_blocks_num = 2
                self.offload_block_cuda_buffers = WeightModuleList([ZImageTransformerBlock(i, self.task, self.mm_type, self.config, True, False, "layers") for i in range(self.offload_blocks_num)])
                self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)
                self.offload_phase_cuda_buffers = None


class ZImageTransformerBlock(WeightModule):
    """
    Z-Image single stream transformer block.
    Based on ZImageTransformerBlock with attention_norm1, attention_norm2, ffn_norm1, ffn_norm2.
    """

    def __init__(
        self,
        layer_id,
        task,
        mm_type,
        config,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        block_prefix="layers",
        modulation=True,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.modulation = modulation
        self.quant_method = config.get("quant_method", None)
        self.sparge = config.get("sparge", False)
        self.ln_type = config.get("ln_type", "Triton")
        self.rms_norm_type = config.get("rms_norm_type", "sgl-kernel")

        self.lazy_load = self.config.get("lazy_load", False)
        if self.lazy_load:
            lazy_load_path = os.path.join(self.config["dit_quantized_ckpt"], f"block_{layer_id}.safetensors")
            self.lazy_load_file = safe_open(lazy_load_path, framework="pt", device="cpu")
        else:
            self.lazy_load_file = None

        # Attention normalization layers
        self.add_module(
            "attention_norm1",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{layer_id}.attention_norm1.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
            ),
        )
        self.add_module(
            "attention_norm2",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{layer_id}.attention_norm2.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
            ),
        )

        # Single stream attention
        self.attention = ZImageSingleStreamAttention(
            layer_id=layer_id,
            block_prefix=block_prefix,
            task=task,
            mm_type=mm_type,
            config=config,
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=create_cpu_buffer,
            lazy_load=self.lazy_load,
            lazy_load_file=self.lazy_load_file,
        )
        self.add_module("attention", self.attention)

        # FFN normalization layers
        # Note: In Z-Image, ffn_norm1 and ffn_norm2 are directly under layers.{layer_id}, not under feed_forward
        self.add_module(
            "ffn_norm1",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{layer_id}.ffn_norm1.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
            ),
        )
        self.add_module(
            "ffn_norm2",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{layer_id}.ffn_norm2.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
            ),
        )

        # Feed forward network
        self.feed_forward = ZImageFeedForward(
            layer_id=layer_id,
            block_prefix=block_prefix,
            task=task,
            mm_type=mm_type,
            config=config,
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=create_cpu_buffer,
            lazy_load=self.lazy_load,
            lazy_load_file=self.lazy_load_file,
        )
        self.add_module("feed_forward", self.feed_forward)

        # AdaLN modulation (if modulation is enabled)
        if self.modulation:
            dim = config["dim"]
            adaln_embed_dim = min(dim, 256)  # ADALN_EMBED_DIM = 256
            self.add_module(
                "adaLN_modulation",
                MM_WEIGHT_REGISTER[self.mm_type](
                    f"{block_prefix}.{layer_id}.adaLN_modulation.0.weight",
                    f"{block_prefix}.{layer_id}.adaLN_modulation.0.bias",
                    create_cuda_buffer,
                    create_cpu_buffer,
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            )


class ZImageSingleStreamAttention(WeightModule):
    """
    Single stream attention for Z-Image.
    Based on ZSingleStreamAttnProcessor.
    """

    def __init__(self, layer_id, block_prefix, task, mm_type, config, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file):
        super().__init__()
        self.layer_id = layer_id
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.quant_method = config.get("quant_method", None)
        self.sparge = config.get("sparge", False)
        self.attn_type = config.get("attn_type", "flash_attn3")
        self.heads = config["n_heads"]
        self.rms_norm_type = config.get("rms_norm_type", "sgl-kernel")

        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        # QK normalization (applied in processor)
        self.add_module(
            "norm_q",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{layer_id}.attention.norm_q.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
            ),
        )
        self.add_module(
            "norm_k",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{block_prefix}.{layer_id}.attention.norm_k.weight",
                create_cuda_buffer=create_cuda_buffer,
                create_cpu_buffer=create_cpu_buffer,
            ),
        )

        # QKV projections
        # Note: Z-Image attention layers don't have bias
        self.add_module(
            "to_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{layer_id}.attention.to_q.weight",
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
                f"{block_prefix}.{layer_id}.attention.to_k.weight",
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
                f"{block_prefix}.{layer_id}.attention.to_v.weight",
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
                        f"{block_prefix}.{layer_id}.attention.to_out.0.weight",
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


class ZImageFeedForward(WeightModule):
    """
    Feed forward network for Z-Image.
    Based on FeedForward with w1, w2, w3 and SiLU gating.
    """

    def __init__(self, layer_id, block_prefix, task, mm_type, config, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file):
        super().__init__()
        self.layer_id = layer_id
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.quant_method = config.get("quant_method", None)
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        dim = config["dim"]
        hidden_dim = int(dim / 3 * 8)  # FeedForward hidden_dim = dim / 3 * 8

        # w1, w2, w3 for SiLU gating
        # Note: Z-Image feed_forward layers don't have bias
        self.add_module(
            "w1",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{layer_id}.feed_forward.w1.weight",
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
                f"{block_prefix}.{layer_id}.feed_forward.w2.weight",
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
                f"{block_prefix}.{layer_id}.feed_forward.w3.weight",
                None,
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

    def forward(self, x):
        w1_out = F.linear(x, self.w1.weight.t(), None)  # Z-Image FFN has no bias
        w3_out = F.linear(x, self.w3.weight.t(), None)
        silu_gated = F.silu(w1_out) * w3_out
        output = F.linear(silu_gated, self.w2.weight.t(), None)
        return output

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)
