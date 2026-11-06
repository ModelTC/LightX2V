# Copyright (c) Meta Platforms, Inc. and affiliates.
# See lightx2v/models/networks/vggt/LICENSE.txt.
# ViT layer portions retain the Apache-2.0 licensing of the source implementation.

from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from lightx2v.common.ops.inference_module import InferenceAttention
from lightx2v.models.networks.vggt.utils.position import PositionGetter


class LayerScaleWeights(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-05, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbedWeights(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])
        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


class RotaryPositionEmbedding2DWeights(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}


class MlpWeights(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer: Callable[..., nn.Module] = nn.GELU, drop: float = 0.0, bias: bool = True
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)


class AttentionWeights(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, proj_bias=True, norm_layer=nn.LayerNorm, qk_norm=False, rope=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = rope
        self.attention = InferenceAttention("torch_sdpa")


class BlockWeights(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=AttentionWeights,
        ffn_layer=MlpWeights,
        qk_norm=False,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, qk_norm=qk_norm, rope=rope)
        self.ls1 = LayerScaleWeights(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, bias=ffn_bias)
        self.ls2 = LayerScaleWeights(dim, init_values=init_values) if init_values else nn.Identity()


class DinoVisionTransformerWeights(nn.Module):
    """The ViT-L/14 patch encoder from the released VGGT-1B checkpoint."""

    def __init__(self, img_size=518, patch_size=14):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedWeights(img_size=img_size, patch_size=patch_size, embed_dim=1024)
        self.cls_token = nn.Parameter(torch.empty(1, 1, 1024))
        self.pos_embed = nn.Parameter(torch.empty(1, self.patch_embed.num_patches + 1, 1024))
        self.register_tokens = nn.Parameter(torch.empty(1, 4, 1024))
        self.blocks = nn.ModuleList([BlockWeights(1024, num_heads=16, norm_layer=partial(nn.LayerNorm, eps=1e-06), init_values=1.0) for _ in range(24)])
        self.norm = nn.LayerNorm(1024, eps=1e-06)


class AggregatorWeights(nn.Module):
    """Alternate frame and global attention, retaining the four DPT feature levels."""

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = DinoVisionTransformerWeights(img_size=img_size, patch_size=patch_size)
        self.rope = RotaryPositionEmbedding2DWeights(frequency=100)
        self.position_getter = PositionGetter()
        self.frame_blocks = nn.ModuleList([BlockWeights(embed_dim, num_heads=16, init_values=0.01, qk_norm=True, rope=self.rope) for _ in range(24)])
        self.global_blocks = nn.ModuleList([BlockWeights(embed_dim, num_heads=16, init_values=0.01, qk_norm=True, rope=self.rope) for _ in range(24)])
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, 4, embed_dim))
        self.patch_start_idx = 5
        self.cached_layers = {4, 11, 17, 23}
        for name, values in (("_resnet_mean", [0.485, 0.456, 0.406]), ("_resnet_std", [0.229, 0.224, 0.225])):
            self.register_buffer(name, torch.tensor(values).view(1, 1, 3, 1, 1), persistent=False)


class CameraHeadWeights(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
    ):
        super().__init__()
        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.trunk = nn.Sequential(*[BlockWeights(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values) for _ in range(trunk_depth)])
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-06)
        self.pose_branch = MlpWeights(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)


class DPTHeadWeights(nn.Module):
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 14.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
        down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
    ) -> None:
        super(DPTHeadWeights, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels])
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        head_features_1 = features
        head_features_2 = 32
        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            conv2_in_channels = head_features_1 // 2
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0)
            )


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlockWeights(features, nn.ReLU(inplace=True), deconv=False, bn=False, expand=False, align_corners=True, size=size, has_residual=has_residual, groups=groups)


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch


class ResidualConvUnitWeights(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init."""
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()


class FeatureFusionBlockWeights(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None, has_residual=True, groups=1):
        """Init."""
        super(FeatureFusionBlockWeights, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups)
        if has_residual:
            self.resConfUnit1 = ResidualConvUnitWeights(features, activation, bn, groups=self.groups)
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnitWeights(features, activation, bn, groups=self.groups)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size


class VGGTWeights(nn.Module):
    """Checkpoint layout for VGGT-1B camera, depth and point-map weights."""

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        self.aggregator = AggregatorWeights(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHeadWeights(dim_in=2 * embed_dim)
        self.point_head = DPTHeadWeights(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHeadWeights(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
