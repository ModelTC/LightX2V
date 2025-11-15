from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import (
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
)


class HunyuanVideo15TransformerWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.task = config["task"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.double_blocks_num = config["mm_double_blocks_depth"]

        self.add_module("double_blocks", WeightModuleList([MMDoubleStreamBlock(i, self.task, self.mm_type, self.config, block_prefix="double_blocks") for i in range(self.double_blocks_num)]))
        self.add_module("final_layer", FinalLayerWeights(self.mm_type))


class MMDoubleStreamBlock(WeightModule):
    def __init__(self, block_index, task, mm_type, config, block_prefix="double_blocks"):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config

        self.lazy_load = False
        self.lazy_load_file = None

        self.add_module(
            "img_branch",
            MMDoubleStreamBlockImgBranch(block_index, task, mm_type, config, block_prefix),
        )
        self.add_module(
            "txt_branch",
            MMDoubleStreamBlockTxtBranch(block_index, task, mm_type, config, block_prefix),
        )


class MMDoubleStreamBlockImgBranch(WeightModule):
    def __init__(self, block_index, task, mm_type, config, block_prefix="double_blocks"):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config

        self.lazy_load = False
        self.lazy_load_file = None

        self.add_module(
            "img_mod",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.img_mod.linear.weight",
                f"{block_prefix}.{self.block_index}.img_mod.linear.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "img_norm1",
            LN_WEIGHT_REGISTER["Default"](
                None,
                None,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "img_attn_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.img_attn_q.weight",
                f"{block_prefix}.{self.block_index}.img_attn_q.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "img_attn_k",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.img_attn_k.weight",
                f"{block_prefix}.{self.block_index}.img_attn_k.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "img_attn_v",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.img_attn_v.weight",
                f"{block_prefix}.{self.block_index}.img_attn_v.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "img_attn_q_norm",
            RMS_WEIGHT_REGISTER["Default"](
                f"{block_prefix}.{self.block_index}.img_attn_q_norm.weight",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "img_attn_k_norm",
            RMS_WEIGHT_REGISTER["Default"](
                f"{block_prefix}.{self.block_index}.img_attn_k_norm.weight",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "img_attn_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.img_attn_proj.weight",
                f"{block_prefix}.{self.block_index}.img_attn_proj.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "img_norm2",
            LN_WEIGHT_REGISTER["Default"](
                None,
                None,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "img_mlp_fc1",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.img_mlp.fc1.weight",
                f"{block_prefix}.{self.block_index}.img_mlp.fc1.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "img_mlp_fc2",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.img_mlp.fc2.weight",
                f"{block_prefix}.{self.block_index}.img_mlp.fc2.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )


class MMDoubleStreamBlockTxtBranch(WeightModule):
    def __init__(self, block_index, task, mm_type, config, block_prefix="double_blocks"):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config

        self.lazy_load = False
        self.lazy_load_file = None

        self.add_module(
            "txt_mod",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.txt_mod.linear.weight",
                f"{block_prefix}.{self.block_index}.txt_mod.linear.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "txt_norm1",
            LN_WEIGHT_REGISTER["Default"](
                None,
                None,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "txt_attn_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.txt_attn_q.weight",
                f"{block_prefix}.{self.block_index}.txt_attn_q.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "txt_attn_k",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.txt_attn_k.weight",
                f"{block_prefix}.{self.block_index}.txt_attn_k.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "txt_attn_v",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.txt_attn_v.weight",
                f"{block_prefix}.{self.block_index}.txt_attn_v.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "txt_attn_q_norm",
            RMS_WEIGHT_REGISTER["Default"](
                f"{block_prefix}.{self.block_index}.txt_attn_q_norm.weight",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "txt_attn_k_norm",
            RMS_WEIGHT_REGISTER["Default"](
                f"{block_prefix}.{self.block_index}.txt_attn_k_norm.weight",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "txt_attn_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.txt_attn_proj.weight",
                f"{block_prefix}.{self.block_index}.txt_attn_proj.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "txt_norm2",
            LN_WEIGHT_REGISTER["Default"](
                None,
                None,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "txt_mlp_fc1",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.txt_mlp.fc1.weight",
                f"{block_prefix}.{self.block_index}.txt_mlp.fc1.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "txt_mlp_fc2",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.txt_mlp.fc2.weight",
                f"{block_prefix}.{self.block_index}.txt_mlp.fc2.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )


class FinalLayerWeights(WeightModule):
    def __init__(self, mm_type):
        super().__init__()
        self.mm_type = mm_type
        self.lazy_load = False
        self.lazy_load_file = None

        self.add_module(
            "adaLN_modulation",
            MM_WEIGHT_REGISTER[self.mm_type](
                "final_layer.adaLN_modulation.1.weight",
                "final_layer.adaLN_modulation.1.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "linear",
            MM_WEIGHT_REGISTER[self.mm_type](
                "final_layer.linear.weight",
                "final_layer.linear.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "norm_final",
            LN_WEIGHT_REGISTER["Default"](
                None,
                None,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
