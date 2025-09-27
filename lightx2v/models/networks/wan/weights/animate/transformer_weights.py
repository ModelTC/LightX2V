import os

from safetensors import safe_open

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class WanAnimateTransformerWeights(WanTransformerWeights):
    def __init__(self, config):
        super().__init__(config)
        self.adapter_blocks_num = self.blocks_num // 5
        for i in range(self.blocks_num):
            if i % 5 == 0:
                self.blocks[i].compute_phases.append(WanAnimateFuserBlock(self.config, i // 5, "face_adapter.fuser_blocks", self.mm_type))
            else:
                self.blocks[i].compute_phases.append(WeightModule())


class WanAnimateFuserBlock(WeightModule):
    def __init__(self, config, block_index, block_prefix, mm_type):
        super().__init__()
        self.config = config
        lazy_load = config.get("lazy_load", False)
        if lazy_load:
            lazy_load_path = os.path.join(config.dit_quantized_ckpt, f"{block_prefix[:-1]}_{block_index}.safetensors")
            lazy_load_file = safe_open(lazy_load_path, framework="pt", device="cpu")
        else:
            lazy_load_file = None

        self.add_module(
            "linear1_kv",
            MM_WEIGHT_REGISTER[mm_type](f"{block_prefix}.{block_index}.linear1_kv.weight", f"{block_prefix}.{block_index}.linear1_kv.bias", lazy_load, lazy_load_file),
        )

        self.add_module(
            "linear1_q",
            MM_WEIGHT_REGISTER[mm_type](f"{block_prefix}.{block_index}.linear1_q.weight", f"{block_prefix}.{block_index}.linear1_q.bias", lazy_load, lazy_load_file),
        )
        self.add_module(
            "linear2",
            MM_WEIGHT_REGISTER[mm_type](f"{block_prefix}.{block_index}.linear2.weight", f"{block_prefix}.{block_index}.linear2.bias", lazy_load, lazy_load_file),
        )

        self.add_module(
            "q_norm",
            RMS_WEIGHT_REGISTER["sgl-kernel"](
                f"{block_prefix}.{block_index}.q_norm.weight",
                lazy_load,
                lazy_load_file,
            ),
        )

        self.add_module(
            "k_norm",
            RMS_WEIGHT_REGISTER["sgl-kernel"](
                f"{block_prefix}.{block_index}.k_norm.weight",
                lazy_load,
                lazy_load_file,
            ),
        )

        self.add_module(
            "pre_norm_feat",
            LN_WEIGHT_REGISTER["Default"](),
        )
        self.add_module(
            "pre_norm_motion",
            LN_WEIGHT_REGISTER["Default"](),
        )

        self.add_module("adapter_attn", ATTN_WEIGHT_REGISTER[config["adapter_attn_type"]]())
