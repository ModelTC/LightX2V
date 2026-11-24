# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from typing import Optional, Tuple  # noqa: UP035 - exported aliases used by the split infer modules

import torch
import torch.distributed as dist
from einops import rearrange
from torch import Tensor, nn

from ..infer.attention import attention
from ..utils.loaders import load_object
from ..utils.type_converter import get_module_device
from .bricks import get_activation_layer, get_norm_layer
from .encoders import MLP, MLPEncoder, TimestepEmbeddingEncoder
from .modulate_layers import ModulateDiT, apply_gate, modulate
from .positional_encoding import RotaryEmbedding


class MMBaseBlock(nn.Module):
    def __init__(self, feat_dim: int, num_heads: int, mlp_ratio: float, dropout: float, positional_encoding_cfg: dict, apply_rope_to_single_branch: bool):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        assert self.feat_dim % num_heads == 0, f"feat_dim {self.feat_dim} must be divisible by num_heads {num_heads}"
        self.head_dim = self.feat_dim // num_heads
        self.mlp_hidden_dim = int(self.feat_dim * mlp_ratio)
        self._positional_encoding_cfg = positional_encoding_cfg.copy()
        self.rotary_emb = RotaryEmbedding(num_feats=self.head_dim, **self._positional_encoding_cfg)
        self.apply_rope_to_single_branch = apply_rope_to_single_branch


class MMDoubleStreamBlock(MMBaseBlock):
    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mlp_act_type: str,
        qk_norm_type: str | None = None,
        qkv_bias: bool = False,
        positional_encoding_cfg: dict | None = None,
        apply_rope_to_single_branch: bool = True,
    ):
        if positional_encoding_cfg is None:
            positional_encoding_cfg = {"max_seq_len": 5000, "use_real": True}
        super().__init__(feat_dim, num_heads, mlp_ratio, dropout, positional_encoding_cfg, apply_rope_to_single_branch)
        self.motion_mod = ModulateDiT(self.feat_dim, factor=6, act_type="silu")
        self.motion_norm1 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-06)
        motion_qkv_out_dim = self.feat_dim * 3
        self.motion_qkv = nn.Linear(self.feat_dim, motion_qkv_out_dim, bias=qkv_bias)
        self.motion_q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-06)
        self.motion_k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-06)
        self.motion_out_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=qkv_bias)
        self.motion_norm2 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-06)
        self.motion_mlp = MLP(self.feat_dim, self.mlp_hidden_dim, act_type=mlp_act_type, bias=True)
        self.text_mod = ModulateDiT(self.feat_dim, factor=6, act_type="silu")
        self.text_norm1 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-06)
        text_qkv_out_dim = self.feat_dim * 3
        self.text_qkv = nn.Linear(self.feat_dim, text_qkv_out_dim, bias=qkv_bias)
        self.text_q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-06)
        self.text_k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-06)
        self.text_out_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=qkv_bias)
        self.text_norm2 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-06)
        self.text_mlp = MLP(self.feat_dim, self.mlp_hidden_dim, act_type=mlp_act_type, bias=True)

    def forward(self, motion_feat: Tensor, text_feat: Tensor, adapter: Tensor, attn_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        from lightx2v.models.networks.hymotion.infer.hymotion_mmdit import MMDoubleStreamBlock_forward

        return MMDoubleStreamBlock_forward(self, motion_feat, text_feat, adapter, attn_mask)


class MMSingleStreamBlock(MMBaseBlock):
    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mlp_act_type: str,
        qk_norm_type: str | None = None,
        qkv_bias: bool = False,
        positional_encoding_cfg: dict | None = None,
        apply_rope_to_single_branch: bool = True,
    ):
        if positional_encoding_cfg is None:
            positional_encoding_cfg = {"max_seq_len": 5000, "use_real": True}
        super().__init__(feat_dim, num_heads, mlp_ratio, dropout, positional_encoding_cfg, apply_rope_to_single_branch)
        self.modulation = ModulateDiT(self.feat_dim, factor=3, act_type="silu")
        self.norm = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-06)
        qkv_factor = 3
        self.linear1 = nn.Linear(self.feat_dim, self.feat_dim * qkv_factor + self.mlp_hidden_dim, bias=qkv_bias)
        self.linear2 = nn.Linear(self.feat_dim + self.mlp_hidden_dim, self.feat_dim, bias=qkv_bias)
        self.q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-06)
        self.k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-06)
        self.mlp_act = get_activation_layer(mlp_act_type)()

    def forward(self, x: Tensor, split_len: int, adapter: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.hymotion_mmdit import MMSingleStreamBlock_forward

        return MMSingleStreamBlock_forward(self, x, split_len, adapter, attn_mask)


class HunyuanMotionMMDiT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feat_dim: int,
        output_dim: int | None = None,
        ctxt_input_dim: int = 4096,
        vtxt_input_dim: int = 256,
        text_refiner_module: str = "hymotion/network/token_refiner.SingleTokenRefiner",
        text_refiner_cfg: dict | None = None,
        num_layers: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        norm_type: str = "layer",
        qk_norm_type: str = "rms",
        qkv_bias: bool = True,
        dropout: float = 0.0,
        final_layer_module: str = "hymotion/network/encoders.FinalLayer",
        final_layer_cfg: dict | None = None,
        mask_mode: str | None = None,
        apply_rope_to_single_branch: bool = True,
        insert_start_token: bool = False,
        with_long_skip_connection: bool = False,
        time_factor: float = 1.0,
        narrowband_length: float = 2.0,
        **kwargs,
    ):
        if text_refiner_cfg is None:
            text_refiner_cfg = {"num_layers": 2}
        if final_layer_cfg is None:
            final_layer_cfg = {"act_type": "silu"}
        super().__init__()
        self.motion_input_dim = input_dim
        self.ctxt_input_dim = ctxt_input_dim
        self.vtxt_input_dim = vtxt_input_dim
        self.feat_dim = feat_dim
        self.output_dim = output_dim or input_dim
        self.mask_mode = mask_mode
        self.insert_start_token = insert_start_token
        self.time_factor = time_factor
        self.narrowband_length = narrowband_length * 30.0
        if self.insert_start_token:
            self.start_token = nn.Parameter(torch.randn(1, feat_dim))
        self.with_long_skip_connection = with_long_skip_connection
        if self.with_long_skip_connection:
            from .encoders import FinalLayer

            self.long_skip_net = FinalLayer(feat_dim=feat_dim, out_dim=feat_dim, act_type="silu")
        self.input_encoder = nn.Linear(in_features=input_dim, out_features=feat_dim)
        self.ctxt_encoder = nn.Linear(in_features=ctxt_input_dim, out_features=feat_dim)
        self.vtxt_encoder = MLPEncoder(in_dim=vtxt_input_dim, feat_dim=feat_dim, num_layers=2, act_type="silu")
        self.timestep_encoder = TimestepEmbeddingEncoder(embedding_dim=feat_dim, feat_dim=feat_dim, time_factor=time_factor)
        if text_refiner_module != "" and text_refiner_module is not None:
            text_refiner_cfg.update(input_dim=feat_dim, feat_dim=feat_dim, num_heads=num_heads)
            self._text_refiner_cfg = text_refiner_cfg.copy()
            self.text_refiner = load_object(text_refiner_module, text_refiner_cfg)
        self.num_layers = num_layers
        assert num_layers % 3 == 0, f"num_layers must be divisible by 3, but got {num_layers}"
        self.mm_double_blocks_layers = int(num_layers // 3)
        self.mm_single_blocks_layers = int(num_layers - num_layers // 3)
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    feat_dim=feat_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    mlp_act_type=mlp_act_type,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    apply_rope_to_single_branch=apply_rope_to_single_branch,
                )
                for _ in range(self.mm_double_blocks_layers)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    feat_dim=feat_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    mlp_act_type=mlp_act_type,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    apply_rope_to_single_branch=apply_rope_to_single_branch,
                )
                for _ in range(self.mm_single_blocks_layers)
            ]
        )
        final_layer_cfg.update(feat_dim=feat_dim, out_dim=self.output_dim)
        self._final_layer_cfg = final_layer_cfg.copy()
        self.final_layer = load_object(final_layer_module, final_layer_cfg)

    def forward(self, x: Tensor, ctxt_input: Tensor, vtxt_input: Tensor, timesteps: Tensor, x_mask_temporal: Tensor, ctxt_mask_temporal: Tensor, **kwargs) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.hymotion_mmdit import HunyuanMotionMMDiT_forward

        return HunyuanMotionMMDiT_forward(self, x, ctxt_input, vtxt_input, timesteps, x_mask_temporal, ctxt_mask_temporal, **kwargs)

    @staticmethod
    def _canonical_mask(input_mask: Tensor) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.hymotion_mmdit import HunyuanMotionMMDiT__canonical_mask

        return HunyuanMotionMMDiT__canonical_mask(input_mask)

    def _build_dmm_attn_mask_shared(self, bsz: int, motion_len: int, text_len: int, dtype: torch.dtype, key_padding_mask: Tensor | None, attn_mask: Tensor | None, device: torch.device) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.hymotion_mmdit import HunyuanMotionMMDiT__build_dmm_attn_mask_shared

        return HunyuanMotionMMDiT__build_dmm_attn_mask_shared(self, bsz, motion_len, text_len, dtype, key_padding_mask, attn_mask, device)

    def _build_smm_attn_mask_shared(self, bsz: int, split_len: int, total_len: int, dtype: torch.dtype, key_padding_mask: Tensor | None, attn_mask: Tensor | None, device: torch.device) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.hymotion_mmdit import HunyuanMotionMMDiT__build_smm_attn_mask_shared

        return HunyuanMotionMMDiT__build_smm_attn_mask_shared(self, bsz, split_len, total_len, dtype, key_padding_mask, attn_mask, device)


__all__ = ["Optional", "Tensor", "Tuple", "apply_gate", "attention", "dist", "get_module_device", "modulate", "rearrange", "torch"]
