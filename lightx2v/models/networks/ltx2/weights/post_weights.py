"""
Post-weights module for LTX2 transformer model.

This module handles the output processing weights including:
- Scale-shift table
- Output normalization
- Output projection
"""

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import (
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    TENSOR_REGISTER,
)


class LTX2PostWeights(WeightModule):
    """
    Post-weights module for LTX2 transformer.

    Handles all weights after transformer blocks:
    - Video output processing (scale_shift_table, norm_out, proj_out)
    - Audio output processing (if audio is enabled)
    """

    def __init__(self, config):
        """
        Initialize post-weights module.

        Args:
            config: Model configuration dictionary containing:
                - model_type: LTXModelType (AudioVideo, VideoOnly, AudioOnly)
                - inner_dim: Video inner dimension
                - audio_inner_dim: Audio inner dimension (if audio enabled)
                - out_channels: Video output channels
                - audio_out_channels: Audio output channels (if audio enabled)
        """
        super().__init__()
        self.config = config

        self.add_module(
            "scale_shift_table",
            TENSOR_REGISTER["Default"](
                "model.diffusion_model.scale_shift_table",
            ),
        )
        self.add_module("norm_out", LN_WEIGHT_REGISTER["Default"]())
        self.add_module(
            "proj_out",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.proj_out.weight",
                "model.diffusion_model.proj_out.bias",
            ),
        )

        self.add_module(
            "audio_scale_shift_table",
            TENSOR_REGISTER["Default"](
                "model.diffusion_model.audio_scale_shift_table",
            ),
        )
        self.add_module("audio_norm_out", LN_WEIGHT_REGISTER["Default"]())
        self.add_module(
            "audio_proj_out",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.audio_proj_out.weight",
                "model.diffusion_model.audio_proj_out.bias",
            ),
        )
