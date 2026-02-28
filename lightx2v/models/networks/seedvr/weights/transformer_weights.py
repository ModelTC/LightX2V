from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, TENSOR_REGISTER


class SeedVRTransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        self.config = config
        self.blocks_num = config["num_layers"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.rms_norm_type = config.get("rms_norm_type", "torch")
        self.norm_eps = config.get("norm_eps", 1.0e-5)
        self.qk_bias = config.get("qk_bias", False)
        self.mlp_type = config.get("mlp_type", "swiglu")

        mm_layers = config.get("mm_layers", 0)
        window = config.get("window")
        window_method = config.get("window_method")
        if window is None or isinstance(window[0], int):
            window = [window] * self.blocks_num
        if window_method is None or isinstance(window_method, str):
            window_method = [window_method] * self.blocks_num

        blocks = WeightModuleList(
            SeedVRTransformerBlockWeights(
                block_index=i,
                shared_weights=not ((i < mm_layers) if isinstance(mm_layers, int) else mm_layers[i]),
                is_last_layer=(i == self.blocks_num - 1),
                mm_type=self.mm_type,
                rms_norm_type=self.rms_norm_type,
                norm_eps=self.norm_eps,
                qk_bias=self.qk_bias,
                mlp_type=self.mlp_type,
                window=window[i],
                window_method=window_method[i],
            )
            for i in range(self.blocks_num)
        )
        self.add_module("blocks", blocks)


class SeedVRTransformerBlockWeights(WeightModule):
    def __init__(
        self,
        *,
        block_index: int,
        shared_weights: bool,
        is_last_layer: bool,
        mm_type: str,
        rms_norm_type: str,
        norm_eps: float,
        qk_bias: bool,
        mlp_type: str,
        window,
        window_method,
    ):
        super().__init__()
        self.block_index = block_index
        self.shared_weights = shared_weights
        self.vid_only = is_last_layer
        self.window = window
        self.window_method = window_method
        self.mlp_type = mlp_type
        self.norm_eps = norm_eps
        self.rms_norm_type = rms_norm_type

        branches = ["all"] if shared_weights else ["vid", "txt"]
        self.branches = branches

        for branch in branches:
            # Attention projections
            qkv_bias_name = f"blocks.{block_index}.attn.proj_qkv.{branch}.bias" if qk_bias else None
            self.add_module(
                f"attn_qkv_{branch}",
                MM_WEIGHT_REGISTER[mm_type](
                    f"blocks.{block_index}.attn.proj_qkv.{branch}.weight",
                    qkv_bias_name,
                ),
            )
            self.add_module(
                f"attn_out_{branch}",
                MM_WEIGHT_REGISTER[mm_type](
                    f"blocks.{block_index}.attn.proj_out.{branch}.weight",
                    f"blocks.{block_index}.attn.proj_out.{branch}.bias",
                ),
            )

            # QK RMS norms
            self.add_module(
                f"attn_norm_q_{branch}",
                RMS_WEIGHT_REGISTER[rms_norm_type](
                    f"blocks.{block_index}.attn.norm_q.{branch}.weight",
                    eps=norm_eps,
                ),
            )
            self.add_module(
                f"attn_norm_k_{branch}",
                RMS_WEIGHT_REGISTER[rms_norm_type](
                    f"blocks.{block_index}.attn.norm_k.{branch}.weight",
                    eps=norm_eps,
                ),
            )

            # MLP
            if mlp_type == "swiglu":
                self.add_module(
                    f"mlp_proj_in_gate_{branch}",
                    MM_WEIGHT_REGISTER[mm_type](
                        f"blocks.{block_index}.mlp.{branch}.proj_in_gate.weight",
                    ),
                )
                self.add_module(
                    f"mlp_proj_in_{branch}",
                    MM_WEIGHT_REGISTER[mm_type](
                        f"blocks.{block_index}.mlp.{branch}.proj_in.weight",
                    ),
                )
                self.add_module(
                    f"mlp_proj_out_{branch}",
                    MM_WEIGHT_REGISTER[mm_type](
                        f"blocks.{block_index}.mlp.{branch}.proj_out.weight",
                    ),
                )
            else:
                self.add_module(
                    f"mlp_proj_in_{branch}",
                    MM_WEIGHT_REGISTER[mm_type](
                        f"blocks.{block_index}.mlp.{branch}.proj_in.weight",
                        f"blocks.{block_index}.mlp.{branch}.proj_in.bias",
                    ),
                )
                self.add_module(
                    f"mlp_proj_out_{branch}",
                    MM_WEIGHT_REGISTER[mm_type](
                        f"blocks.{block_index}.mlp.{branch}.proj_out.weight",
                        f"blocks.{block_index}.mlp.{branch}.proj_out.bias",
                    ),
                )

            # AdaSingle parameters
            self.add_module(
                f"ada_attn_shift_{branch}",
                TENSOR_REGISTER["Default"](f"blocks.{block_index}.ada.{branch}.attn_shift"),
            )
            self.add_module(
                f"ada_attn_scale_{branch}",
                TENSOR_REGISTER["Default"](f"blocks.{block_index}.ada.{branch}.attn_scale"),
            )
            self.add_module(
                f"ada_attn_gate_{branch}",
                TENSOR_REGISTER["Default"](f"blocks.{block_index}.ada.{branch}.attn_gate"),
            )
            self.add_module(
                f"ada_mlp_shift_{branch}",
                TENSOR_REGISTER["Default"](f"blocks.{block_index}.ada.{branch}.mlp_shift"),
            )
            self.add_module(
                f"ada_mlp_scale_{branch}",
                TENSOR_REGISTER["Default"](f"blocks.{block_index}.ada.{branch}.mlp_scale"),
            )
            self.add_module(
                f"ada_mlp_gate_{branch}",
                TENSOR_REGISTER["Default"](f"blocks.{block_index}.ada.{branch}.mlp_gate"),
            )
