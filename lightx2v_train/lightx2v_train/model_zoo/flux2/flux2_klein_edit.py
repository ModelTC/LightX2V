from lightx2v_train.model_zoo.flux2.capability_adapters import Flux2EditDistributionMatchingCapability
from lightx2v_train.utils.registry import MODEL_REGISTER

from .flux2_klein import Flux2KleinModel


@MODEL_REGISTER("flux2_klein_edit")
class Flux2KleinEditModel(Flux2KleinModel):
    """Supports weights from these Hugging Face repos:
    - https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
    - https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B
    """

    requires_source_images = True
    target_latent_mode = "mode"
    distribution_matching_capability_cls = Flux2EditDistributionMatchingCapability
    shared_condition_keys = ("reference_tokens", "reference_ids")
