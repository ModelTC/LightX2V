from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class HunyuanTransformerWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.double_blocks_num = 20
        self.single_blocks_num = 40

        self.add_module("double_blocks", WeightModuleList([HunyuanTransformerDoubleBlock(i, self.config) for i in range(self.double_blocks_num)]))
        self.add_module("single_blocks", WeightModuleList([HunyuanTransformerSingleBlock(i, self.config) for i in range(self.single_blocks_num)]))


class HunyuanTransformerDoubleBlock(WeightModule):
    def __init__(self, block_index, config):
        super().__init__()
        self.block_index = block_index
        self.config = config

        if self.config["do_mm_calib"]:
            mm_type = "Calib"
        else:
            mm_type = self.config["mm_config"].get("mm_type", "Default") if self.config["mm_config"] else "Default"

        self.add_module("img_mod", MM_WEIGHT_REGISTER[mm_type](f"double_blocks.{self.block_index}.img_mod.linear.weight", f"double_blocks.{self.block_index}.img_mod.linear.bias"))
        self.add_module("img_attn_qkv", MM_WEIGHT_REGISTER[mm_type](f"double_blocks.{self.block_index}.img_attn_qkv.weight", f"double_blocks.{self.block_index}.img_attn_qkv.bias"))
        self.add_module("img_attn_q_norm", RMS_WEIGHT_REGISTER["sgl-kernel"](f"double_blocks.{self.block_index}.img_attn_q_norm.weight", eps=1e-6))
        self.add_module("img_attn_k_norm", RMS_WEIGHT_REGISTER["sgl-kernel"](f"double_blocks.{self.block_index}.img_attn_k_norm.weight", eps=1e-6))
        self.add_module("img_attn_proj", MM_WEIGHT_REGISTER[mm_type](f"double_blocks.{self.block_index}.img_attn_proj.weight", f"double_blocks.{self.block_index}.img_attn_proj.bias"))
        self.add_module("img_mlp_fc1", MM_WEIGHT_REGISTER[mm_type](f"double_blocks.{self.block_index}.img_mlp.fc1.weight", f"double_blocks.{self.block_index}.img_mlp.fc1.bias"))
        self.add_module("img_mlp_fc2", MM_WEIGHT_REGISTER[mm_type](f"double_blocks.{self.block_index}.img_mlp.fc2.weight", f"double_blocks.{self.block_index}.img_mlp.fc2.bias"))

        self.add_module("txt_mod", MM_WEIGHT_REGISTER[mm_type](f"double_blocks.{self.block_index}.txt_mod.linear.weight", f"double_blocks.{self.block_index}.txt_mod.linear.bias"))
        self.add_module("txt_attn_qkv", MM_WEIGHT_REGISTER[mm_type](f"double_blocks.{self.block_index}.txt_attn_qkv.weight", f"double_blocks.{self.block_index}.txt_attn_qkv.bias"))
        self.add_module("txt_attn_q_norm", RMS_WEIGHT_REGISTER["sgl-kernel"](f"double_blocks.{self.block_index}.txt_attn_q_norm.weight", eps=1e-6))
        self.add_module("txt_attn_k_norm", RMS_WEIGHT_REGISTER["sgl-kernel"](f"double_blocks.{self.block_index}.txt_attn_k_norm.weight", eps=1e-6))
        self.add_module("txt_attn_proj", MM_WEIGHT_REGISTER[mm_type](f"double_blocks.{self.block_index}.txt_attn_proj.weight", f"double_blocks.{self.block_index}.txt_attn_proj.bias"))
        self.add_module("txt_mlp_fc1", MM_WEIGHT_REGISTER[mm_type](f"double_blocks.{self.block_index}.txt_mlp.fc1.weight", f"double_blocks.{self.block_index}.txt_mlp.fc1.bias"))
        self.add_module("txt_mlp_fc2", MM_WEIGHT_REGISTER[mm_type](f"double_blocks.{self.block_index}.txt_mlp.fc2.weight", f"double_blocks.{self.block_index}.txt_mlp.fc2.bias"))

        # attention weights section
        self.add_module("double_attn", ATTN_WEIGHT_REGISTER[self.config["attention_type"]]())


class HunyuanTransformerSingleBlock(WeightModule):
    def __init__(self, block_index, config):
        super().__init__()
        self.block_index = block_index
        self.config = config
        self.sparge = config.get("sparge", False)

        if self.config["do_mm_calib"]:
            mm_type = "Calib"
        else:
            mm_type = self.config["mm_config"].get("mm_type", "Default") if self.config["mm_config"] else "Default"

        self.add_module("linear1", MM_WEIGHT_REGISTER[mm_type](f"single_blocks.{self.block_index}.linear1.weight", f"single_blocks.{self.block_index}.linear1.bias"))
        self.add_module("linear2", MM_WEIGHT_REGISTER[mm_type](f"single_blocks.{self.block_index}.linear2.weight", f"single_blocks.{self.block_index}.linear2.bias"))
        self.add_module("q_norm", RMS_WEIGHT_REGISTER["sgl-kernel"](f"single_blocks.{self.block_index}.q_norm.weight", eps=1e-6))
        self.add_module("k_norm", RMS_WEIGHT_REGISTER["sgl-kernel"](f"single_blocks.{self.block_index}.k_norm.weight", eps=1e-6))
        self.add_module("modulation", MM_WEIGHT_REGISTER[mm_type](f"single_blocks.{self.block_index}.modulation.linear.weight", f"single_blocks.{self.block_index}.modulation.linear.bias"))

        # attention weights section
        if self.sparge:
            # load sparge attention weights
            #! todo
            pass
        else:
            self.add_module("single_attn", ATTN_WEIGHT_REGISTER[self.config["attention_type"]]())
