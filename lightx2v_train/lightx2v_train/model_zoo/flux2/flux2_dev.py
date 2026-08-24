import torch
from diffusers import Flux2Pipeline

from lightx2v_train.model_capabilities import ConsistencyModelCapability, DistributionMatchingCapability, FlowMatchingSFTCapability
from lightx2v_train.model_zoo.capability_adapters import SpatialLatentGeometry
from lightx2v_train.model_zoo.capability_adapters.common import GenericDistributionMatchingCapability, GenericFlowMatchingCapability
from lightx2v_train.model_zoo.flux2.capability_adapters import Flux2ConsistencyModelCapability
from lightx2v_train.utils.registry import MODEL_REGISTER

from .common import Flux2ModelBase


@MODEL_REGISTER("flux2_dev")
class Flux2DevModel(Flux2ModelBase):
    """Supports weights from these Hugging Face repos:
    - https://huggingface.co/black-forest-labs/FLUX.2-dev
    """

    pipeline_cls = Flux2Pipeline
    distribution_matching_capability_cls = GenericDistributionMatchingCapability
    default_text_encoder_out_layers = (10, 20, 30)

    def register_capabilities(self):
        super().register_capabilities()
        self.capabilities.register(FlowMatchingSFTCapability, GenericFlowMatchingCapability(self))
        self.capabilities.register(
            DistributionMatchingCapability,
            self.distribution_matching_capability_cls(
                self,
                latent_geometry=SpatialLatentGeometry(
                    channels_path="transformer.config.in_channels",
                    spatial_downsample_multiplier=2,
                ),
                guidance_in_denoiser_space=True,
            ),
        )
        self.capabilities.register(ConsistencyModelCapability, Flux2ConsistencyModelCapability(self))

    def _configure_model(self):
        self.guidance_scale = float(self.config["model"].get("guidance_scale", 4.0))

    def _reuse_model_state_from(self, source):
        self.guidance_scale = source.guidance_scale

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        guidance = torch.full(
            (denoiser_input.hidden_states.shape[0],),
            self.guidance_scale,
            device=self.device,
            dtype=torch.float32,
        )
        return self._denoise(denoiser_input, timestep_or_sigma, condition, guidance)

    def apply_cfg(self, positive, negative, guidance_scale):
        del positive, negative, guidance_scale
        raise ValueError("Flux2 Dev uses embedded guidance and does not support external classifier-free guidance")
