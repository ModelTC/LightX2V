# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

from lightx2v.models.networks.trellis2.infer.transformer_infer import Trellis2TransformerInfer
from lightx2v.models.networks.trellis2.weights.loader import load_component
from lightx2v.models.networks.trellis2.weights.transformer_weights import SLatFlowModelWeights, SparseStructureFlowModelWeights


class Trellis2FlowModel:
    """Connect a loaded flow checkpoint to explicit inference."""

    def __init__(self, weights):
        self.weights = weights
        self.in_channels = weights.in_channels
        self.resolution = weights.resolution
        self.transformer_infer = Trellis2TransformerInfer()

    def __call__(self, x, t, cond, **kwargs):
        if isinstance(self.weights, SparseStructureFlowModelWeights):
            return self.transformer_infer.infer_dense_flow(self.weights, x, t, cond, **kwargs)
        return self.transformer_infer.infer_sparse_flow(self.weights, x, t, cond, **kwargs)


def load_trellis2_component(*args, **kwargs):
    weights = load_component(*args, **kwargs)
    if isinstance(weights, (SparseStructureFlowModelWeights, SLatFlowModelWeights)):
        return Trellis2FlowModel(weights)
    return weights
