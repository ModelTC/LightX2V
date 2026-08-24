"""Distribution-matching capability for Qwen-Image-Edit."""

import torch

from lightx2v_train.model_zoo.capability_adapters.common import (
    GenericDistributionMatchingCapability,
    _cached_condition_pair,
    _negative_prompt,
    _require_single_prompt,
)


class QwenImageEditDistributionMatchingCapability(GenericDistributionMatchingCapability):
    """Keep source-image conditioning in Qwen-Image-Edit DMD."""

    cache_uses_sample_context = True

    def encode_conditions(self, batch, negative_prompt, guidance_scale, broadcast):
        conditioning = batch["conditioning"]
        prompt = conditioning.get("prompt", "")
        scalar = _require_single_prompt(prompt)
        cached = _cached_condition_pair(batch, self.model, require_negative=guidance_scale > 1)
        if cached is not None:
            positive, negative = cached
            return broadcast(positive), broadcast(negative) if negative is not None else None
        prompts = [prompt]
        if guidance_scale > 1:
            prompts.append(_negative_prompt(conditioning, negative_prompt, scalar=scalar))

        with torch.no_grad():
            conditions = self.model.encode_conditions_with_source(batch, prompts)
        positive = conditions[0]
        negative = conditions[1] if len(conditions) > 1 else None
        return broadcast(positive), broadcast(negative) if negative is not None else None
