from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanFFN,
    WanSelfAttention,
)
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
    TENSOR_REGISTER,
)


class WanMtxg3TransformerWeights(WeightModule):
    """Transformer weights for Matrix-Game-3.0.

    Each block has:
    - SelfAttention (reused from base)
    - CrossAttention (with norm3 / cross_attn_norm support)
    - Camera injection layers (cam_injector_layer1/2, cam_scale_layer, cam_shift_layer)
    - ActionModule (keyboard_embed, mouse_mlp, mouse/keyboard cross-attn, only on specified blocks)
    - FFN (reused from base)
    """

    def __init__(self, config):
        super().__init__()
        self.blocks_num = config["num_layers"]
        self.task = config["task"]
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        if self.mm_type != "Default":
            assert config.get("dit_quantized") is True

        action_config = config.get("action_config", {})
        action_blocks = action_config.get("blocks", [])

        block_list = []
        for i in range(self.blocks_num):
            has_action = i in action_blocks
            block_list.append(WanMtxg3TransformerBlock(i, self.task, self.mm_type, self.config, has_action=has_action))
        self.blocks = WeightModuleList(block_list)
        self.add_module("blocks", self.blocks)

        # Non-block weights (head)
        self.register_parameter("norm", LN_WEIGHT_REGISTER["torch"]())
        self.add_module("head", MM_WEIGHT_REGISTER["Default"]("head.head.weight", "head.head.bias"))
        self.register_parameter("head_modulation", TENSOR_REGISTER["Default"]("head.modulation"))

    def non_block_weights_to_cuda(self):
        self.norm.to_cuda()
        self.head.to_cuda()
        self.head_modulation.to_cuda()

    def non_block_weights_to_cpu(self):
        self.norm.to_cpu()
        self.head.to_cpu()
        self.head_modulation.to_cpu()


class WanMtxg3TransformerBlock(WeightModule):
    """Single transformer block for MG3.0.

    Phases:
    0: SelfAttention
    1: CamInjection (per-block camera plucker scale/shift)
    2: CrossAttention
    3: ActionModule (only on action blocks; None placeholder otherwise)
    4: FFN
    """

    def __init__(self, block_index, task, mm_type, config, has_action=False, block_prefix="blocks"):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.has_action = has_action

        phases = [
            WanSelfAttention(block_index, block_prefix, task, mm_type, config),
            WanMtxg3CamInjection(block_index, block_prefix, mm_type, config),
            WanMtxg3CrossAttention(block_index, block_prefix, task, mm_type, config),
        ]
        if has_action:
            phases.append(WanMtxg3ActionModule(block_index, block_prefix, task, mm_type, config))
        phases.append(WanFFN(block_index, block_prefix, task, mm_type, config))

        self.compute_phases = WeightModuleList(phases)
        self.add_module("compute_phases", self.compute_phases)


class WanMtxg3CamInjection(WeightModule):
    """Per-block camera plucker injection weights.

    From the official MG3 WanAttentionBlock:
        cam_injector_layer1, cam_injector_layer2: SiLU + residual MLP
        cam_scale_layer, cam_shift_layer: scale/shift modulation
    """

    def __init__(self, block_index, block_prefix, mm_type, config):
        super().__init__()
        self.block_index = block_index

        self.add_module(
            "cam_injector_layer1",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.cam_injector_layer1.weight",
                f"{block_prefix}.{block_index}.cam_injector_layer1.bias",
            ),
        )
        self.add_module(
            "cam_injector_layer2",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.cam_injector_layer2.weight",
                f"{block_prefix}.{block_index}.cam_injector_layer2.bias",
            ),
        )
        self.add_module(
            "cam_scale_layer",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.cam_scale_layer.weight",
                f"{block_prefix}.{block_index}.cam_scale_layer.bias",
            ),
        )
        self.add_module(
            "cam_shift_layer",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.cam_shift_layer.weight",
                f"{block_prefix}.{block_index}.cam_shift_layer.bias",
            ),
        )


class WanMtxg3CrossAttention(WeightModule):
    """Cross-attention weights for MG3.0 blocks.

    Same as base WanCrossAttention but without the image encoder K/V/norm_k_img
    branches (MG3.0 does not use a separate image encoder cross-attention).
    Also includes norm3 with elementwise_affine=True (cross_attn_norm=True in MG3.0).
    """

    def __init__(self, block_index, block_prefix, task, mm_type, config):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.config = config

        if self.config.get("sf_config", False):
            self.attn_rms_norm_type = self.config.get("rms_norm_type", "self_forcing")
        else:
            self.attn_rms_norm_type = self.config.get("rms_norm_type", "sgl-kernel")

        # norm3 with elementwise_affine=True for cross_attn_norm
        self.add_module(
            "norm3",
            LN_WEIGHT_REGISTER["torch"](
                f"{block_prefix}.{block_index}.norm3.weight",
                f"{block_prefix}.{block_index}.norm3.bias",
            ),
        )
        self.add_module(
            "cross_attn_q",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.cross_attn.q.weight",
                f"{block_prefix}.{block_index}.cross_attn.q.bias",
            ),
        )
        self.add_module(
            "cross_attn_k",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.cross_attn.k.weight",
                f"{block_prefix}.{block_index}.cross_attn.k.bias",
            ),
        )
        self.add_module(
            "cross_attn_v",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.cross_attn.v.weight",
                f"{block_prefix}.{block_index}.cross_attn.v.bias",
            ),
        )
        self.add_module(
            "cross_attn_o",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.cross_attn.o.weight",
                f"{block_prefix}.{block_index}.cross_attn.o.bias",
            ),
        )
        self.add_module(
            "cross_attn_norm_q",
            RMS_WEIGHT_REGISTER[self.attn_rms_norm_type](
                f"{block_prefix}.{block_index}.cross_attn.norm_q.weight",
            ),
        )
        self.add_module(
            "cross_attn_norm_k",
            RMS_WEIGHT_REGISTER[self.attn_rms_norm_type](
                f"{block_prefix}.{block_index}.cross_attn.norm_k.weight",
            ),
        )
        self.add_module("cross_attn_1", ATTN_WEIGHT_REGISTER[self.config["cross_attn_1_type"]]())


class WanMtxg3ActionModule(WeightModule):
    """ActionModule weights for MG3.0 blocks.

    From the official MG3 ActionModule:
    - keyboard_embed (2-layer MLP with SiLU)
    - mouse_mlp (Linear -> GELU -> Linear -> LayerNorm)
    - t_qkv (combined QKV projection for mouse)
    - proj_mouse, proj_keyboard (output projections)
    - mouse_attn_q, keyboard_attn_kv (keyboard cross-attn projections)
    - img_attn_q_norm, img_attn_k_norm, key_attn_q_norm, key_attn_k_norm (RMSNorm)
    """

    def __init__(self, block_index, block_prefix, task, mm_type, config):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.config = config

        attn_rms_norm_type = config.get("rms_norm_type", "sgl-kernel")

        # Keyboard embed (2-layer MLP)
        self.add_module(
            "keyboard_embed_0",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.action_model.keyboard_embed.0.weight",
                f"{block_prefix}.{block_index}.action_model.keyboard_embed.0.bias",
            ),
        )
        self.add_module(
            "keyboard_embed_2",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.action_model.keyboard_embed.2.weight",
                f"{block_prefix}.{block_index}.action_model.keyboard_embed.2.bias",
            ),
        )

        # Mouse MLP
        self.add_module(
            "mouse_mlp_0",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.action_model.mouse_mlp.0.weight",
                f"{block_prefix}.{block_index}.action_model.mouse_mlp.0.bias",
            ),
        )
        self.add_module(
            "mouse_mlp_2",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.action_model.mouse_mlp.2.weight",
                f"{block_prefix}.{block_index}.action_model.mouse_mlp.2.bias",
            ),
        )
        self.add_module(
            "mouse_mlp_3",
            LN_WEIGHT_REGISTER["torch"](
                f"{block_prefix}.{block_index}.action_model.mouse_mlp.3.weight",
                f"{block_prefix}.{block_index}.action_model.mouse_mlp.3.bias",
                eps=1e-5,
            ),
        )

        # Mouse QKV and projection
        self.add_module(
            "t_qkv",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.action_model.t_qkv.weight",
                bias_name=None,
            ),
        )
        self.add_module(
            "proj_mouse",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.action_model.proj_mouse.weight",
                bias_name=None,
            ),
        )

        # Mouse attention RMS norms
        self.add_module(
            "img_attn_q_norm",
            RMS_WEIGHT_REGISTER[attn_rms_norm_type](
                f"{block_prefix}.{block_index}.action_model.img_attn_q_norm.weight",
            ),
        )
        self.add_module(
            "img_attn_k_norm",
            RMS_WEIGHT_REGISTER[attn_rms_norm_type](
                f"{block_prefix}.{block_index}.action_model.img_attn_k_norm.weight",
            ),
        )

        # Keyboard cross-attn
        self.add_module(
            "mouse_attn_q",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.action_model.mouse_attn_q.weight",
                bias_name=None,
            ),
        )
        self.add_module(
            "keyboard_attn_kv",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.action_model.keyboard_attn_kv.weight",
                bias_name=None,
            ),
        )
        self.add_module(
            "proj_keyboard",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.action_model.proj_keyboard.weight",
                bias_name=None,
            ),
        )

        # Keyboard attention RMS norms
        self.add_module(
            "key_attn_q_norm",
            RMS_WEIGHT_REGISTER[attn_rms_norm_type](
                f"{block_prefix}.{block_index}.action_model.key_attn_q_norm.weight",
            ),
        )
        self.add_module(
            "key_attn_k_norm",
            RMS_WEIGHT_REGISTER[attn_rms_norm_type](
                f"{block_prefix}.{block_index}.action_model.key_attn_k_norm.weight",
            ),
        )

        # Flash attention module for action cross-attn
        self.add_module("action_attn", ATTN_WEIGHT_REGISTER[self.config.get("cross_attn_1_type", "flash_attn2")]())
