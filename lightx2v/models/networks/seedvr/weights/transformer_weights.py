import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.seedvr.utils import rope as _seedvr_rope  # noqa: F401
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, ROPE_REGISTER, TENSOR_REGISTER


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

        window = config.get("window")
        window_method = config.get("window_method")
        if window is None or isinstance(window[0], int):
            window = [window] * self.blocks_num
        if window_method is None or isinstance(window_method, str):
            window_method = [window_method] * self.blocks_num

        self.window = window
        self.window_method = window_method
        self.mm_layers = config.get("mm_layers", self.blocks_num)
        self.last_layer_vid_only = bool(config.get("last_layer_vid_only"))

        self.register_offload_buffers()
        blocks = WeightModuleList(self._make_block(i) for i in range(self.blocks_num))
        self.add_module("blocks", blocks)

    def _block_uses_shared_weights(self, block_index):
        is_mm_block = block_index < self.mm_layers if isinstance(self.mm_layers, int) else self.mm_layers[block_index]
        return not is_mm_block

    def _make_block(self, block_index, *, create_cuda_buffer=False, branches=None, alias_shared_to_vid=False):
        return SeedVRTransformerBlockWeights(
            config=self.config,
            block_index=block_index,
            shared_weights=self._block_uses_shared_weights(block_index),
            vid_only=(self.last_layer_vid_only and block_index == self.blocks_num - 1),
            mm_type=self.mm_type,
            rms_norm_type=self.rms_norm_type,
            norm_eps=self.norm_eps,
            qk_bias=self.qk_bias,
            mlp_type=self.mlp_type,
            window=self.window[block_index],
            window_method=self.window_method[block_index],
            create_cuda_buffer=create_cuda_buffer,
            branches=branches,
            alias_shared_to_vid=alias_shared_to_vid,
        )

    def register_offload_buffers(self):
        self.offload_block_cuda_buffers = None
        self.offload_phase_cuda_buffers = None
        if not self.config.get("cpu_offload", False) or self.config.get("offload_granularity", "block") != "block":
            return

        split_block_indices = [i for i in range(self.blocks_num) if not self._block_uses_shared_weights(i)]
        if split_block_indices:
            template_index = split_block_indices[0]
            branches = ("vid", "txt")
            alias_shared_to_vid = len(split_block_indices) != self.blocks_num
        else:
            template_index = 0
            branches = ("all",)
            alias_shared_to_vid = False

        self.offload_block_cuda_buffers = WeightModuleList(
            self._make_block(
                template_index,
                create_cuda_buffer=True,
                branches=branches,
                alias_shared_to_vid=alias_shared_to_vid,
            )
            for _ in range(2)
        )
        self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)


class SeedVRTransformerBlockWeights(WeightModule):
    def __init__(
        self,
        *,
        config,
        block_index: int,
        shared_weights: bool,
        vid_only: bool,
        mm_type: str,
        rms_norm_type: str,
        norm_eps: float,
        qk_bias: bool,
        mlp_type: str,
        window,
        window_method,
        create_cuda_buffer: bool = False,
        branches=None,
        alias_shared_to_vid: bool = False,
    ):
        super().__init__()
        self.config = config
        self.block_index = block_index
        self.shared_weights = shared_weights
        self.vid_only = vid_only
        self.window = window
        self.window_method = window_method
        self.mlp_type = mlp_type
        self.norm_eps = norm_eps
        self.rms_norm_type = rms_norm_type
        self.create_cuda_buffer = create_cuda_buffer
        self.alias_shared_to_vid = alias_shared_to_vid
        self.add_module(
            "rope",
            ROPE_REGISTER[config.get("rope_type", "rope3d")](layout="interleaved", compute_dtype=torch.float32),
        )
        self.rope.set_config(config)

        self.branches = list(branches) if branches is not None else (["all"] if shared_weights else ["vid", "txt"])

        for branch in self.branches:
            # Attention projections
            qkv_bias_name = f"blocks.{block_index}.attn.proj_qkv.{branch}.bias" if qk_bias else None
            self.add_module(
                f"attn_qkv_{branch}",
                MM_WEIGHT_REGISTER[mm_type](
                    f"blocks.{block_index}.attn.proj_qkv.{branch}.weight",
                    qkv_bias_name,
                    create_cuda_buffer=create_cuda_buffer,
                ),
            )
            self.add_module(
                f"attn_out_{branch}",
                MM_WEIGHT_REGISTER[mm_type](
                    f"blocks.{block_index}.attn.proj_out.{branch}.weight",
                    f"blocks.{block_index}.attn.proj_out.{branch}.bias",
                    create_cuda_buffer=create_cuda_buffer,
                ),
            )

            # QK RMS norms
            self.add_module(
                f"attn_norm_q_{branch}",
                RMS_WEIGHT_REGISTER[rms_norm_type](
                    f"blocks.{block_index}.attn.norm_q.{branch}.weight",
                    create_cuda_buffer=create_cuda_buffer,
                    eps=norm_eps,
                ),
            )
            self.add_module(
                f"attn_norm_k_{branch}",
                RMS_WEIGHT_REGISTER[rms_norm_type](
                    f"blocks.{block_index}.attn.norm_k.{branch}.weight",
                    create_cuda_buffer=create_cuda_buffer,
                    eps=norm_eps,
                ),
            )

            # MLP
            if mlp_type == "swiglu":
                self.add_module(
                    f"mlp_proj_in_gate_{branch}",
                    MM_WEIGHT_REGISTER[mm_type](
                        f"blocks.{block_index}.mlp.{branch}.proj_in_gate.weight",
                        None,
                        create_cuda_buffer=create_cuda_buffer,
                    ),
                )
                self.add_module(
                    f"mlp_proj_in_{branch}",
                    MM_WEIGHT_REGISTER[mm_type](
                        f"blocks.{block_index}.mlp.{branch}.proj_in.weight",
                        None,
                        create_cuda_buffer=create_cuda_buffer,
                    ),
                )
                self.add_module(
                    f"mlp_proj_out_{branch}",
                    MM_WEIGHT_REGISTER[mm_type](
                        f"blocks.{block_index}.mlp.{branch}.proj_out.weight",
                        None,
                        create_cuda_buffer=create_cuda_buffer,
                    ),
                )
            else:
                self.add_module(
                    f"mlp_proj_in_{branch}",
                    MM_WEIGHT_REGISTER[mm_type](
                        f"blocks.{block_index}.mlp.{branch}.proj_in.weight",
                        f"blocks.{block_index}.mlp.{branch}.proj_in.bias",
                        create_cuda_buffer=create_cuda_buffer,
                    ),
                )
                self.add_module(
                    f"mlp_proj_out_{branch}",
                    MM_WEIGHT_REGISTER[mm_type](
                        f"blocks.{block_index}.mlp.{branch}.proj_out.weight",
                        f"blocks.{block_index}.mlp.{branch}.proj_out.bias",
                        create_cuda_buffer=create_cuda_buffer,
                    ),
                )

            # AdaSingle parameters
            self.add_module(
                f"ada_attn_shift_{branch}",
                TENSOR_REGISTER["Default"](f"blocks.{block_index}.ada.{branch}.attn_shift", create_cuda_buffer=create_cuda_buffer),
            )
            self.add_module(
                f"ada_attn_scale_{branch}",
                TENSOR_REGISTER["Default"](f"blocks.{block_index}.ada.{branch}.attn_scale", create_cuda_buffer=create_cuda_buffer),
            )
            self.add_module(
                f"ada_attn_gate_{branch}",
                TENSOR_REGISTER["Default"](f"blocks.{block_index}.ada.{branch}.attn_gate", create_cuda_buffer=create_cuda_buffer),
            )
            self.add_module(
                f"ada_mlp_shift_{branch}",
                TENSOR_REGISTER["Default"](f"blocks.{block_index}.ada.{branch}.mlp_shift", create_cuda_buffer=create_cuda_buffer),
            )
            self.add_module(
                f"ada_mlp_scale_{branch}",
                TENSOR_REGISTER["Default"](f"blocks.{block_index}.ada.{branch}.mlp_scale", create_cuda_buffer=create_cuda_buffer),
            )
            self.add_module(
                f"ada_mlp_gate_{branch}",
                TENSOR_REGISTER["Default"](f"blocks.{block_index}.ada.{branch}.mlp_gate", create_cuda_buffer=create_cuda_buffer),
            )

        if self.alias_shared_to_vid:
            module_names = ["attn_qkv", "attn_out", "attn_norm_q", "attn_norm_k", "mlp_proj_in", "mlp_proj_out"]
            if self.mlp_type == "swiglu":
                module_names.append("mlp_proj_in_gate")
            tensor_names = ["ada_attn_shift", "ada_attn_scale", "ada_attn_gate", "ada_mlp_shift", "ada_mlp_scale", "ada_mlp_gate"]
            for name in module_names + tensor_names:
                setattr(self, f"{name}_all", getattr(self, f"{name}_vid"))

    def _set_runtime_metadata(self, block_index):
        mm_layers = self.config.get("mm_layers", self.config["num_layers"])
        is_mm_block = block_index < mm_layers if isinstance(mm_layers, int) else mm_layers[block_index]
        self.block_index = block_index
        self.shared_weights = not is_mm_block
        self.vid_only = bool(self.config.get("last_layer_vid_only")) and block_index == self.config["num_layers"] - 1

        window = self.config.get("window")
        self.window = window if window is None or isinstance(window[0], int) else window[block_index]
        window_method = self.config.get("window_method")
        self.window_method = window_method if window_method is None or isinstance(window_method, str) else window_method[block_index]

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        if not self.create_cuda_buffer:
            return super().load_state_dict(destination, block_index, adapter_block_index)

        self._set_runtime_metadata(block_index)
        if self.shared_weights and self.alias_shared_to_vid:
            destination = {(name.replace(".all.", ".vid.") if ".all." in name else name): tensor for name, tensor in destination.items()}
        return super().load_state_dict(destination, block_index, adapter_block_index)
