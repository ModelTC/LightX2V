from lightx2v_train.model_zoo.flux2.capability_adapters import Flux2EditDistributionMatchingCapability
from lightx2v_train.utils.registry import MODEL_REGISTER

from .flux2_dev import Flux2DevModel


@MODEL_REGISTER("flux2_dev_edit")
class Flux2DevEditModel(Flux2DevModel):
    requires_source_images = True
    target_latent_mode = "mode"
    distribution_matching_capability_cls = Flux2EditDistributionMatchingCapability
