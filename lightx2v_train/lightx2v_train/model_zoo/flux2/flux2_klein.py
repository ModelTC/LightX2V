from diffusers import Flux2KleinPipeline

from lightx2v_train.model_capabilities import (
    AdapterBankCapability,
    ConsistencyModelCapability,
    DistributionMatchingCapability,
    DopsdCapability,
    FlowMatchingSFTCapability,
)
from lightx2v_train.model_zoo.capability_adapters import PeftAdapterBankCapability, SpatialLatentGeometry
from lightx2v_train.model_zoo.capability_adapters.common import GenericDistributionMatchingCapability, GenericFlowMatchingCapability
from lightx2v_train.model_zoo.flux2.capability_adapters import Flux2ConsistencyModelCapability, Flux2DopsdCapability
from lightx2v_train.utils.registry import MODEL_REGISTER

from .common import Flux2ModelBase


@MODEL_REGISTER("flux2_klein")
class Flux2KleinModel(Flux2ModelBase):
    """Flux2 Klein Base model. Step-distilled Klein checkpoints are unsupported."""

    pipeline_cls = Flux2KleinPipeline
    distribution_matching_capability_cls = GenericDistributionMatchingCapability
    default_text_encoder_out_layers = (9, 18, 27)
    supports_dopsd = True

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
            ),
        )
        self.capabilities.register(ConsistencyModelCapability, Flux2ConsistencyModelCapability(self))
        if self.supports_dopsd:
            self.capabilities.register(DopsdCapability, Flux2DopsdCapability(self))
        self.capabilities.register(AdapterBankCapability, PeftAdapterBankCapability(self))

    def _validate_model_path(self, model_path):
        pipeline_config = self.pipeline_cls.load_config(model_path)
        if pipeline_config.get("is_distilled", False):
            raise ValueError("Flux2KleinModel supports only FLUX.2-klein-base-4B/9B checkpoints; step-distilled FLUX.2-klein-4B/9B checkpoints are unsupported")

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        return self._denoise(denoiser_input, timestep_or_sigma, condition, guidance=None)

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return self.pipeline_cls._unpack_latents_with_ids(
            prediction,
            denoiser_input.target_ids,
            height=denoiser_input.height,
            width=denoiser_input.width,
        )
