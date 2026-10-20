# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from lightx2v.common.ops.inference_module import InferenceAttention, bind_linear_operators, cast_matrix_weights
from lightx2v.common.ops.sparse import SparseConv3d
from lightx2v.common.ops.sparse.linear import bind_sparse_linear_operators
from lightx2v.common.ops.sparse.neighbor_map import NEIGHBOR_MAPS
from lightx2v.models.networks.trellis2.sparse_structure_flow import SparseStructureFlowModel
from lightx2v.models.networks.trellis2.structured_latent_flow import SLatFlowModel
from lightx2v.models.video_encoders.hf.trellis2.autoencoders.fdg_vae import FlexiDualGridVaeDecoder
from lightx2v.models.video_encoders.hf.trellis2.autoencoders.sparse_structure_vae import SparseStructureDecoder
from lightx2v.models.video_encoders.hf.trellis2.autoencoders.sparse_unet_vae import SparseUnetVaeDecoder

MODEL_TYPES = {
    "SparseStructureFlowModel": SparseStructureFlowModel,
    "SLatFlowModel": SLatFlowModel,
    "SparseStructureDecoder": SparseStructureDecoder,
    "FlexiDualGridVaeDecoder": FlexiDualGridVaeDecoder,
    "SparseUnetVaeDecoder": SparseUnetVaeDecoder,
}


def load_component(path, device, neighbor_map="sorted", attention_backend="torch_sdpa", linear_backend="Default"):
    path = Path(path)
    config = json.loads(path.with_suffix(".json").read_text())
    with torch.device("meta"):
        architecture = {key: value for (key, value) in config["args"].items() if key not in {"use_checkpoint", "initialization", "use_fp16", "dtype"}}
        model = MODEL_TYPES[config["name"]](**architecture)
    model.load_state_dict(load_file(str(path.with_suffix(".safetensors"))), strict=True, assign=True)
    model.eval().to(device)
    cast_matrix_weights(model, extra_types=(SparseConv3d,))
    bind_linear_operators(model, linear_backend)
    bind_sparse_linear_operators(model, linear_backend)
    for layer in model.modules():
        if isinstance(layer, SparseConv3d):
            layer.neighbor_map = NEIGHBOR_MAPS[neighbor_map]
        elif isinstance(layer, InferenceAttention):
            layer.op = InferenceAttention(attention_backend).op
    return model
