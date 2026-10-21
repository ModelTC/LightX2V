from pathlib import Path

import torch

from lightx2v.common.ops.inference_module import InferenceAttention, bind_linear_operators, cast_matrix_weights
from lightx2v.models.networks.vggt.layers.attention import Attention
from lightx2v.models.networks.vggt.model import VGGT


def load_vggt(model_path, device, attention_backend="torch_sdpa", linear_backend="Default"):
    checkpoint = Path(model_path) / "model.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # Tracking is outside the reconstruction API; do not allocate its unused head.
    state = {key: value for key, value in state.items() if not key.startswith("track_head.") and key != "aggregator.patch_embed.mask_token"}
    with torch.device("meta"):
        model = VGGT()
    model.load_state_dict(state, strict=True, assign=True)
    # Normalization buffers are non-persistent and are not in the checkpoint.
    for name, value in (("_resnet_mean", [0.485, 0.456, 0.406]), ("_resnet_std", [0.229, 0.224, 0.225])):
        setattr(model.aggregator, name, torch.tensor(value).reshape(1, 1, 3, 1, 1))
    model.eval().to(device)
    cast_matrix_weights(model)
    bind_linear_operators(model, linear_backend)
    for layer in model.modules():
        if isinstance(layer, Attention):
            layer.attention = InferenceAttention(attention_backend)
    return model
