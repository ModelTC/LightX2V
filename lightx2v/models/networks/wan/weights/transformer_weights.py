import os

from safetensors import safe_open

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
    TENSOR_REGISTER,
)


class WanTransformerWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.blocks_num = config["num_layers"]
        self.task = config["task"]
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        if self.mm_type != "Default":
            assert config.get("dit_quantized") is True
        if config.get("do_mm_calib", False):
            self.mm_type = "Calib"
        self.blocks = WeightModuleList([WanTransformerAttentionBlock(i, self.task, self.mm_type, self.config) for i in range(self.blocks_num)])
        self.add_module("blocks", self.blocks)

        # non blocks weights
        self.register_parameter("norm", LN_WEIGHT_REGISTER["Default"]())
        self.add_module("head", MM_WEIGHT_REGISTER["Default"]("head.head.weight", "head.head.bias"))
        self.register_parameter("head_modulation", TENSOR_REGISTER["Default"]("head.modulation"))

    def clear(self):
        for block in self.blocks:
            for phase in block.compute_phases:
                phase.clear()

    def non_block_weights_to_cuda(self):
        self.norm.to_cuda()
        self.head.to_cuda()
        self.head_modulation.to_cuda()

    def non_block_weights_to_cpu(self):
        self.norm.to_cpu()
        self.head.to_cpu()
        self.head_modulation.to_cpu()


class WanTransformerAttentionBlock(WeightModule):
    def __init__(self, block_index, task, mm_type, config, block_prefix="blocks"):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.quant_method = config.get("quant_method", None)

        self.lazy_load = self.config.get("lazy_load", False)
        if self.lazy_load:
            lazy_load_path = os.path.join(self.config["dit_quantized_ckpt"], f"block_{block_index}.safetensors")
            self.lazy_load_file = safe_open(lazy_load_path, framework="pt", device="cpu")
        else:
            self.lazy_load_file = None

        self.compute_phases = WeightModuleList(
            [
                WanSelfAttention(
                    block_index,
                    block_prefix,
                    task,
                    mm_type,
                    config,
                    self.lazy_load,
                    self.lazy_load_file,
                ),
                WanCrossAttention(
                    block_index,
                    block_prefix,
                    task,
                    mm_type,
                    config,
                    self.lazy_load,
                    self.lazy_load_file,
                ),
                WanFFN(
                    block_index,
                    block_prefix,
                    task,
                    mm_type,
                    config,
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            ]
        )

        self.add_module("compute_phases", self.compute_phases)


class WanSelfAttention(WeightModule):
    def __init__(self, block_index, block_prefix, task, mm_type, config, lazy_load, lazy_load_file):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.quant_method = config.get("quant_method", None)

        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        if self.config.get("sf_config", False):
            self.attn_rms_type = "self_forcing"
        else:
            self.attn_rms_type = "sgl-kernel"

        self.add_module(
            "modulation",
            TENSOR_REGISTER["Default"](
                f"{block_prefix}.{self.block_index}.modulation",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        self.add_module(
            "norm1",
            LN_WEIGHT_REGISTER["Default"](),
        )

        self.add_module(
            "self_attn_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.self_attn.q.weight",
                f"{block_prefix}.{self.block_index}.self_attn.q.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        self.add_module(
            "self_attn_k",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.self_attn.k.weight",
                f"{block_prefix}.{self.block_index}.self_attn.k.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "self_attn_v",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.self_attn.v.weight",
                f"{block_prefix}.{self.block_index}.self_attn.v.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "self_attn_o",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.self_attn.o.weight",
                f"{block_prefix}.{self.block_index}.self_attn.o.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "self_attn_norm_q",
            RMS_WEIGHT_REGISTER[self.attn_rms_type](
                f"{block_prefix}.{self.block_index}.self_attn.norm_q.weight",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "self_attn_norm_k",
            RMS_WEIGHT_REGISTER[self.attn_rms_type](
                f"{block_prefix}.{self.block_index}.self_attn.norm_k.weight",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        attention_weights_cls = ATTN_WEIGHT_REGISTER[self.config["self_attn_1_type"]]
        if self.config["self_attn_1_type"] == "svg_attn":
            attention_weights_cls.prepare(
                head_num=self.config["num_heads"],
                head_dim=self.config["dim"] // self.config["num_heads"],
                sample_mse_max_row=self.config.get("svg_sample_mse_max_row", 10000),
                num_sampled_rows=self.config.get("svg_num_sampled_rows", 64),
                context_length=self.config.get("svg_context_length", 0),
                sparsity=self.config.get("svg_sparsity", 0.25),
            )
        if self.config["self_attn_1_type"] in ["svg_attn", "radial_attn", "nbhd_attn", "nbhd_attn_flashinfer"]:
            attention_weights_cls.attnmap_frame_num = self.config["attnmap_frame_num"]
        # nbhd_attn setting
        if self.config["self_attn_1_type"] in ["nbhd_attn", "nbhd_attn_flashinfer"]:
            if "nbhd_attn_setting" in self.config:
                if "coefficient" in self.config["nbhd_attn_setting"]:
                    attention_weights_cls.coefficient = self.config["nbhd_attn_setting"]["coefficient"]
                if "min_width" in self.config["nbhd_attn_setting"]:
                    attention_weights_cls.min_width = self.config["nbhd_attn_setting"]["min_width"]
        self.add_module("self_attn_1", attention_weights_cls())

        if self.config["seq_parallel"]:
            self.add_module("self_attn_1_parallel", ATTN_WEIGHT_REGISTER[self.config["parallel"].get("seq_p_attn_type", "ulysses")]())

        if self.quant_method in ["advanced_ptq"]:
            self.add_module(
                "smooth_norm1_weight",
                TENSOR_REGISTER["Default"](
                    f"{block_prefix}.{self.block_index}.affine_norm1.weight",
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            )
            self.add_module(
                "smooth_norm1_bias",
                TENSOR_REGISTER["Default"](
                    f"{block_prefix}.{self.block_index}.affine_norm1.bias",
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            )


class WanCrossAttention(WeightModule):
    def __init__(self, block_index, block_prefix, task, mm_type, config, lazy_load, lazy_load_file):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        if self.config.get("sf_config", False):
            self.attn_rms_type = "self_forcing"
        else:
            self.attn_rms_type = "sgl-kernel"

        self.add_module(
            "norm3",
            LN_WEIGHT_REGISTER["Default"](
                f"{block_prefix}.{self.block_index}.norm3.weight",
                f"{block_prefix}.{self.block_index}.norm3.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "cross_attn_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.cross_attn.q.weight",
                f"{block_prefix}.{self.block_index}.cross_attn.q.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "cross_attn_k",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.cross_attn.k.weight",
                f"{block_prefix}.{self.block_index}.cross_attn.k.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "cross_attn_v",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.cross_attn.v.weight",
                f"{block_prefix}.{self.block_index}.cross_attn.v.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "cross_attn_o",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.cross_attn.o.weight",
                f"{block_prefix}.{self.block_index}.cross_attn.o.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "cross_attn_norm_q",
            RMS_WEIGHT_REGISTER[self.attn_rms_type](
                f"{block_prefix}.{self.block_index}.cross_attn.norm_q.weight",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "cross_attn_norm_k",
            RMS_WEIGHT_REGISTER[self.attn_rms_type](
                f"{block_prefix}.{self.block_index}.cross_attn.norm_k.weight",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module("cross_attn_1", ATTN_WEIGHT_REGISTER[self.config["cross_attn_1_type"]]())

        if self.config["task"] in ["i2v", "flf2v", "animate", "s2v"] and self.config.get("use_image_encoder", True):
            self.add_module(
                "cross_attn_k_img",
                MM_WEIGHT_REGISTER[self.mm_type](
                    f"{block_prefix}.{self.block_index}.cross_attn.k_img.weight",
                    f"{block_prefix}.{self.block_index}.cross_attn.k_img.bias",
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            )
            self.add_module(
                "cross_attn_v_img",
                MM_WEIGHT_REGISTER[self.mm_type](
                    f"{block_prefix}.{self.block_index}.cross_attn.v_img.weight",
                    f"{block_prefix}.{self.block_index}.cross_attn.v_img.bias",
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            )
            self.add_module(
                "cross_attn_norm_k_img",
                RMS_WEIGHT_REGISTER[self.attn_rms_type](
                    f"{block_prefix}.{self.block_index}.cross_attn.norm_k_img.weight",
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            )
            self.add_module("cross_attn_2", ATTN_WEIGHT_REGISTER[self.config["cross_attn_2_type"]]())


class WanFFN(WeightModule):
    def __init__(self, block_index, block_prefix, task, mm_type, config, lazy_load, lazy_load_file):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.quant_method = config.get("quant_method", None)
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        self.add_module(
            "norm2",
            LN_WEIGHT_REGISTER["Default"](),
        )

        self.add_module(
            "ffn_0",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.ffn.0.weight",
                f"{block_prefix}.{self.block_index}.ffn.0.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "ffn_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.ffn.2.weight",
                f"{block_prefix}.{self.block_index}.ffn.2.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        if self.quant_method in ["advanced_ptq"]:
            self.add_module(
                "smooth_norm2_weight",
                TENSOR_REGISTER["Default"](
                    f"{block_prefix}.{self.block_index}.affine_norm3.weight",
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            )
            self.add_module(
                "smooth_norm2_bias",
                TENSOR_REGISTER["Default"](
                    f"{block_prefix}.{self.block_index}.affine_norm3.bias",
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            )
