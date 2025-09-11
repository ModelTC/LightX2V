import os

from safetensors import safe_open

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class QwenImageTransformerWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.blocks_num = config["num_layers"]
        self.task = config["task"]
        self.config = config
        if config["do_mm_calib"]:
            self.mm_type = "Calib"
        else:
            self.mm_type = config["mm_config"].get("mm_type", "Default") if config["mm_config"] else "Default"

        blocks = WeightModuleList(QwenImageTransformerAttentionBlock(i, self.task, self.mm_type, self.config, "transformer_blocks") for i in range(self.blocks_num))
        self.add_module("blocks", blocks)


class QwenImageTransformerAttentionBlock(WeightModule):
    def __init__(self, block_index, task, mm_type, config, block_prefix="transformer_blocks"):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.quant_method = config.get("quant_method", None)
        self.sparge = config.get("sparge", False)

        self.lazy_load = self.config.get("lazy_load", False)
        if self.lazy_load:
            lazy_load_path = os.path.join(self.config.dit_quantized_ckpt, f"block_{block_index}.safetensors")
            self.lazy_load_file = safe_open(lazy_load_path, framework="pt", device="cpu")
        else:
            self.lazy_load_file = None

        # Image processing modules
        self.add_module(
            "img_mod",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.img_mod.1.weight",
                f"{block_prefix}.{self.block_index}.img_mod.1.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "img_norm1",
            LN_WEIGHT_REGISTER["Default"](eps=1e-6),
        )
        self.attn = QwenImageCrossAttention(
            block_index=block_index, block_prefix="transformer_blocks", task=config.task, mm_type=mm_type, config=config, lazy_load=self.lazy_load, lazy_load_file=self.lazy_load_file
        )
        self.add_module("attn", self.attn)

        self.add_module(
            "img_norm2",
            LN_WEIGHT_REGISTER["Default"](eps=1e-6),
        )
        img_mlp = QwenImageFFN(
            block_index=block_index,
            block_prefix="transformer_blocks",
            ffn_prefix="img_mlp",
            task=config.task,
            mm_type=mm_type,
            config=config,
            lazy_load=self.lazy_load,
            lazy_load_file=self.lazy_load_file,
        )
        self.add_module("img_mlp", img_mlp)

        # Text processing modules
        self.add_module(
            "txt_mod",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.txt_mod.1.weight",
                f"{block_prefix}.{self.block_index}.txt_mod.1.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "txt_norm1",
            LN_WEIGHT_REGISTER["Default"](eps=1e-6),
        )

        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.add_module(
            "txt_norm2",
            LN_WEIGHT_REGISTER["Default"](eps=1e-6),
        )
        txt_mlp = QwenImageFFN(
            block_index=block_index,
            block_prefix="transformer_blocks",
            ffn_prefix="txt_mlp",
            task=config.task,
            mm_type=mm_type,
            config=config,
            lazy_load=self.lazy_load,
            lazy_load_file=self.lazy_load_file,
        )
        self.add_module("txt_mlp", txt_mlp)


class QwenImageCrossAttention(WeightModule):
    def __init__(self, block_index, block_prefix, task, mm_type, config, lazy_load, lazy_load_file):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.quant_method = config.get("quant_method", None)
        self.sparge = config.get("sparge", False)

        self.heads = config["attention_out_dim"] // config["attention_dim_head"]

        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        # norm_q
        self.add_module(
            "norm_q",
            RMS_WEIGHT_REGISTER["fp32_variance"](f"{block_prefix}.{block_index}.attn.norm_q.weight"),
        )
        # norm_k
        self.add_module(
            "norm_k",
            RMS_WEIGHT_REGISTER["fp32_variance"](f"{block_prefix}.{block_index}.attn.norm_k.weight"),
        )
        # to_q
        self.add_module(
            "to_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.to_q.weight",
                f"{block_prefix}.{self.block_index}.attn.to_q.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        # to_k
        self.add_module(
            "to_k",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.to_k.weight",
                f"{block_prefix}.{self.block_index}.attn.to_k.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        # to_v
        self.add_module(
            "to_v",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.to_v.weight",
                f"{block_prefix}.{self.block_index}.attn.to_v.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        # add_q_proj
        self.add_module(
            "add_q_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.add_q_proj.weight",
                f"{block_prefix}.{self.block_index}.attn.add_q_proj.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        # add_k_proj
        self.add_module(
            "add_k_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.add_k_proj.weight",
                f"{block_prefix}.{self.block_index}.attn.add_k_proj.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        # add_v_proj
        self.add_module(
            "add_v_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.add_v_proj.weight",
                f"{block_prefix}.{self.block_index}.attn.add_v_proj.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        # to_out
        self.add_module(
            "to_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.to_out.0.weight",
                f"{block_prefix}.{self.block_index}.attn.to_out.0.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        # to_add_out
        self.add_module(
            "to_add_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.attn.to_add_out.weight",
                f"{block_prefix}.{self.block_index}.attn.to_add_out.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        # norm_added_q
        self.add_module(
            "norm_added_q",
            RMS_WEIGHT_REGISTER["fp32_variance"](f"{block_prefix}.{block_index}.attn.norm_added_q.weight"),
        )
        # norm_added_k
        self.add_module(
            "norm_added_k",
            RMS_WEIGHT_REGISTER["fp32_variance"](f"{block_prefix}.{block_index}.attn.norm_added_k.weight"),
        )
        # attn
        self.add_module("torch_sdpa", ATTN_WEIGHT_REGISTER["torch_sdpa"]())

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)


class QwenImageFFN(WeightModule):
    def __init__(self, block_index, block_prefix, ffn_prefix, task, mm_type, config, lazy_load, lazy_load_file):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.quant_method = config.get("quant_method", None)
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        self.add_module(
            "mlp_0",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.{ffn_prefix}.net.0.proj.weight",
                f"{block_prefix}.{self.block_index}.{ffn_prefix}.net.0.proj.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "mlp_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.{ffn_prefix}.net.2.weight",
                f"{block_prefix}.{self.block_index}.{ffn_prefix}.net.2.bias",
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
