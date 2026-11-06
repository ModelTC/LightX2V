# Copyright (c) Meta Platforms, Inc. and affiliates.
# See lightx2v/models/networks/vggt/LICENSE.txt.
# ViT layer portions retain the Apache-2.0 licensing of the source implementation.

from lightx2v.models.networks.vggt.infer.transformer_infer import VGGTTransformerInfer
from lightx2v.models.networks.vggt.weights.loader import load_vggt


class VGGTModel:
    """Connect checkpoint loading to explicit VGGT inference."""

    def __init__(self, config, device):
        self.weights = load_vggt(
            config["model_path"],
            device,
            config.get("attn_type", "torch_sdpa"),
            config.get("dit_quant_scheme", "Default"),
            config.get("ln_norm_type", "torch"),
            config.get("conv_type", "Default"),
        )
        self.transformer_infer = VGGTTransformerInfer()

    def __call__(self, images):
        return self.transformer_infer.infer(self.weights, images)
