# Copyright (c) Meta Platforms, Inc. and affiliates.
# See lightx2v/models/networks/vggt/LICENSE.txt.

from pathlib import Path

import torch

from lightx2v.common.ops.inference_module import InferenceAttention, bind_linear_operators, bind_spatial_operators, cast_matrix_weights
from lightx2v.models.networks.vggt.weights.transformer_weights import AttentionWeights, VGGTWeights


def load_vggt(model_path, device, attention_backend="torch_sdpa", linear_backend="Default", norm_backend="torch", conv_backend="Default"):
    checkpoint = Path(model_path) / "model.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key: value for (key, value) in state.items() if not key.startswith("track_head.") and key != "aggregator.patch_embed.mask_token"}
    with torch.device("meta"):
        model = VGGTWeights()
    model.load_state_dict(state, strict=True, assign=True)
    for name, value in (("_resnet_mean", [0.485, 0.456, 0.406]), ("_resnet_std", [0.229, 0.224, 0.225])):
        setattr(model.aggregator, name, torch.tensor(value).reshape(1, 1, 3, 1, 1))
    model.eval().to(device)
    # The original VGGT runs the aggregator under BF16 autocast but keeps
    # the prediction heads and their checkpoint weights in FP32.
    cast_matrix_weights(model.aggregator)
    bind_linear_operators(model.aggregator, linear_backend)
    bind_spatial_operators(model.aggregator, norm_backend, conv_backend)
    for layer in model.aggregator.modules():
        if isinstance(layer, AttentionWeights):
            layer.attention = InferenceAttention(attention_backend)
    for layer in model.camera_head.modules():
        if isinstance(layer, AttentionWeights):
            layer.attention = InferenceAttention('torch_sdpa')
    return model
