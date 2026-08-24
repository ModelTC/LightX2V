"""Flow-matching SFT capability for Wan-family video models."""

from lightx2v_train.model_zoo.capability_adapters.common import GenericFlowMatchingCapability
from lightx2v_train.model_zoo.wan.training_cache import encode_wan_video_cache


class WanFlowMatchingCapability(GenericFlowMatchingCapability):
    """Use the common SFT objective with Wan's video-latent cache path."""

    def encode_training_cache(self, batch):
        return encode_wan_video_cache(self.model, batch)
