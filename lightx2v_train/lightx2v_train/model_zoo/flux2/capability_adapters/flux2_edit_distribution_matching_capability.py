import torch

from lightx2v_train.model_zoo.capability_adapters.common import (
    GenericDistributionMatchingCapability,
    _negative_prompt,
    _require_single_prompt,
)


class Flux2EditDistributionMatchingCapability(GenericDistributionMatchingCapability):
    """Keep reference-image conditioning in Flux2 edit DMD."""

    def encode_conditions(self, batch, negative_prompt, guidance_scale, broadcast):
        conditioning = batch["conditioning"]
        prompt = conditioning.get("prompt", "")
        scalar = _require_single_prompt(prompt)
        prompts = [prompt]
        if guidance_scale > 1:
            prompts.append(_negative_prompt(conditioning, negative_prompt, scalar=scalar))

        with torch.no_grad():
            conditions = self.model.encode_conditions_with_source(batch, prompts)
        positive = conditions[0]
        negative = conditions[1] if len(conditions) > 1 else None
        return broadcast(positive), broadcast(negative) if negative is not None else None
