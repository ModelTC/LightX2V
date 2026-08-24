from lightx2v_train.model_zoo.flux2.capability_adapters import Flux2EditDistributionMatchingCapability
from lightx2v_train.utils.registry import MODEL_REGISTER

from .flux2_dev import Flux2DevModel


@MODEL_REGISTER("flux2_dev_edit")
class Flux2DevEditModel(Flux2DevModel):
    """Supports weights from these Hugging Face repos:
    - https://huggingface.co/black-forest-labs/FLUX.2-dev
    """

    requires_source_images = True
    target_latent_mode = "mode"
    distribution_matching_capability_cls = Flux2EditDistributionMatchingCapability
    shared_condition_keys = ("reference_tokens", "reference_ids")
