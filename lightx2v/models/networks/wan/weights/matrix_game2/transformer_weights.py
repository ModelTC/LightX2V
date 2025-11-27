import os

from safetensors import safe_open

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanFFN,
    WanSelfAttention,
    WanTransformerAttentionBlock,
)
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
    TENSOR_REGISTER,
)


class WanActionTransformerWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.blocks_num = config["num_layers"]
        self.task = config["task"]
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        if self.mm_type != "Default":
            assert config.get("dit_quantized") is True

        action_blocks = config["action_config"]["blocks"]
        block_list = []
        for i in range(self.blocks_num):
            if i in action_blocks:
                block_list.append(WanTransformerActionBlock(i, self.task, self.mm_type, self.config, "blocks"))
            else:
                block_list.append(WanTransformerAttentionBlock(i, self.task, self.mm_type, self.config, False, "blocks"))
        self.blocks = WeightModuleList(block_list)
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


class WanTransformerActionBlock(WeightModule):
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
                    False,
                    self.lazy_load,
                    self.lazy_load_file,
                ),
                WanActionCrossAttention(
                    block_index,
                    block_prefix,
                    task,
                    mm_type,
                    config,
                    self.lazy_load,
                    self.lazy_load_file,
                ),
                WanActionModule(
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
                    False,
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            ]
        )

        self.add_module("compute_phases", self.compute_phases)


class WanActionModule(WeightModule):
    def __init__(self, block_index, block_prefix, task, mm_type, config, lazy_load, lazy_load_file):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.quant_method = config.get("quant_method", None)

        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        self.attn_rms_type = "self_forcing"

        self.add_module(
            "keyboard_embed_0",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.action_model.keyboard_embed.0.weight",
                f"{block_prefix}.{self.block_index}.action_model.keyboard_embed.0.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module(
            "keyboard_embed_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.action_model.keyboard_embed.2.weight",
                f"{block_prefix}.{self.block_index}.action_model.keyboard_embed.2.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        self.add_module(
            "proj_keyboard",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.action_model.proj_keyboard.weight",
                bias_name=None,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
            ),
        )

        self.add_module(
            "keyboard_attn_kv",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.action_model.keyboard_attn_kv.weight",
                bias_name=None,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
            ),
        )

        self.add_module("cross_attn_2", ATTN_WEIGHT_REGISTER[self.config["cross_attn_2_type"]]())

        self.add_module(
            "mouse_attn_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.action_model.mouse_attn_q.weight",
                bias_name=None,
                lazy_load=self.lazy_load,
                lazy_load_file=self.lazy_load_file,
            ),
        )

        if self.config["mode"] != "templerun":
            self.add_module(
                "t_qkv",
                MM_WEIGHT_REGISTER[self.mm_type](
                    f"{block_prefix}.{self.block_index}.action_model.t_qkv.weight",
                    bias_name=None,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                ),
            )

            self.add_module(
                "proj_mouse",
                MM_WEIGHT_REGISTER[self.mm_type](
                    f"{block_prefix}.{self.block_index}.action_model.proj_mouse.weight",
                    bias_name=None,
                    lazy_load=self.lazy_load,
                    lazy_load_file=self.lazy_load_file,
                ),
            )

            self.add_module(
                "mouse_mlp_0",
                MM_WEIGHT_REGISTER[self.mm_type](
                    f"{block_prefix}.{self.block_index}.action_model.mouse_mlp.0.weight",
                    f"{block_prefix}.{self.block_index}.action_model.mouse_mlp.0.bias",
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            )
            self.add_module(
                "mouse_mlp_2",
                MM_WEIGHT_REGISTER[self.mm_type](
                    f"{block_prefix}.{self.block_index}.action_model.mouse_mlp.2.weight",
                    f"{block_prefix}.{self.block_index}.action_model.mouse_mlp.2.bias",
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            )
            self.add_module(
                "mouse_mlp_3",
                LN_WEIGHT_REGISTER[self.mm_type](
                    f"{block_prefix}.{self.block_index}.action_model.mouse_mlp.3.weight",
                    f"{block_prefix}.{self.block_index}.action_model.mouse_mlp.3.bias",
                    self.lazy_load,
                    self.lazy_load_file,
                    eps=1e-6,
                ),
            )


class WanActionCrossAttention(WeightModule):
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
