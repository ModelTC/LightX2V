# Copyright (c) Microsoft Corporation.
# See lightx2v/models/networks/trellis2/LICENSE.

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from lightx2v.common.ops.inference_module import InferenceAttention, cast_matrix_weights
from lightx2v.common.ops.sparse import SparseConv3d
from lightx2v.common.ops.sparse.neighbor_map import NEIGHBOR_MAPS
from lightx2v.models.networks.trellis2.weights.transformer_weights import SLatFlowModelWeights, SparseStructureFlowModelWeights
from lightx2v.models.video_encoders.hf.trellis2.autoencoders.fdg_vae import FlexiDualGridVaeDecoder
from lightx2v.models.video_encoders.hf.trellis2.autoencoders.sparse_structure_vae import SparseStructureDecoder
from lightx2v.models.video_encoders.hf.trellis2.autoencoders.sparse_unet_vae import SparseUnetVaeDecoder

MODEL_TYPES = {
    "SparseStructureFlowModel": SparseStructureFlowModelWeights,
    "SLatFlowModel": SLatFlowModelWeights,
    "SparseStructureDecoder": SparseStructureDecoder,
    "FlexiDualGridVaeDecoder": FlexiDualGridVaeDecoder,
    "SparseUnetVaeDecoder": SparseUnetVaeDecoder,
}


def load_component(path, device, neighbor_map="sorted", attention_backend="torch_sdpa", linear_backend="Default", norm_backend="torch", conv_backend="Default", sparse_conv_backend="flex_gemm"):
    if linear_backend != "Default" or norm_backend != "torch" or conv_backend != "Default":
        raise NotImplementedError("Precision-aligned TRELLIS.2 requires the original linear, normalization and convolution operators")
    path = Path(path)
    config = json.loads(path.with_suffix(".json").read_text())
    with torch.device("meta"):
        architecture = {key: value for (key, value) in config["args"].items() if key not in {"use_checkpoint", "initialization", "use_fp16", "dtype"}}
        model = MODEL_TYPES[config["name"]](**architecture)
    model.load_state_dict(load_file(str(path.with_suffix(".safetensors"))), strict=True, assign=True)
    # The original graph keeps the input/output projections in FP32 and casts
    # only the transformer or decoder torso to its configured compute dtype.
    model.eval().to(device).float()
    if isinstance(model, (SparseStructureFlowModelWeights, SLatFlowModelWeights)):
        dtype = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                 "float16": torch.float16, "fp16": torch.float16,
                 "float32": torch.float32, "fp32": torch.float32}[config["args"].get("dtype", "float32")]
    else:
        dtype = torch.float16 if config["args"].get("use_fp16", False) else torch.float32
    model.dtype = dtype
    cast_matrix_weights(model.blocks, dtype=dtype, extra_types=(SparseConv3d,))
    if hasattr(model, "middle_block"):
        cast_matrix_weights(model.middle_block, dtype=dtype, extra_types=(SparseConv3d,))
    for layer in model.modules():
        if isinstance(layer, SparseConv3d):
            layer.neighbor_map = NEIGHBOR_MAPS[neighbor_map]
            if sparse_conv_backend not in ("flex_gemm", "torch"):
                raise ValueError(f"Unsupported sparse convolution backend: {sparse_conv_backend}")
            layer.backend = sparse_conv_backend
        elif isinstance(layer, InferenceAttention):
            layer.op = InferenceAttention(attention_backend).op
    return model
