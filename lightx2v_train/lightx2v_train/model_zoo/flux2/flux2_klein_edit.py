from lightx2v_train.model_zoo.flux2.capability_adapters import Flux2EditDistributionMatchingCapability
from lightx2v_train.utils.registry import MODEL_REGISTER

from .flux2_klein import Flux2KleinModel


@MODEL_REGISTER("flux2_klein_edit")
class Flux2KleinEditModel(Flux2KleinModel):
    requires_source_images = True
    target_latent_mode = "mode"
    distribution_matching_capability_cls = Flux2EditDistributionMatchingCapability
    supports_dopsd = False
