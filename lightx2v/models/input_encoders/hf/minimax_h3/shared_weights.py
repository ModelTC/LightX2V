"""Shared CPU source weights for the native H3 Qwen3-VL text prefix."""

import torch

from lightx2v.common.offload.shared_weight_coordinator import coordinate_rank_local_error
from lightx2v.common.offload.shared_weight_map import validate_shared_operator_views
from lightx2v.models.networks.minimax_h3.shared_block_weights import load_h3_shared_weights


def load_shared_text_weights(encoder, backbone, path, text_config):
    def validate():
        if not encoder.block_offload or encoder.tensor_parallel or encoder.config.get("text_encoder_quantized", False):
            raise ValueError("H3 shared text weights require unquantized block offload without text TP")

    expected = {name: (shape, torch.bfloat16) for name, shape in encoder._expected_weight_shapes(text_config).items()}
    weights = load_h3_shared_weights(path, encoder.config, "text", expected=expected, validate=validate)
    error = None
    try:
        backbone.load(weights)
        backbone.to_cpu()
        validate_shared_operator_views(weights, backbone.state_dict())
    except Exception as exc:
        error = exc
    try:
        coordinate_rank_local_error("H3 text operator binding", error)
    except BaseException:
        weights.owner.close()
        raise
    # The source tensors and this owner must outlive every GPU transfer.
    backbone.shared_cpu_weight_owner = weights.owner
