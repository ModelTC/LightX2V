"""Distribution-matching capability for Qwen-Image-Edit."""

import torch

from lightx2v_train.model_zoo.capability_adapters.common import (
    GenericDistributionMatchingCapability,
    _cached_condition_pair,
    _negative_prompt,
    _require_single_prompt,
)
from lightx2v_train.utils.generation_shapes import normalize_generation_shape


class QwenImageEditDistributionMatchingCapability(GenericDistributionMatchingCapability):
    """Keep source-image conditioning in Qwen-Image-Edit DMD."""

    cache_uses_sample_context = True

    def validate_generation_shapes(self, generation_shapes):
        if generation_shapes is not None:
            raise ValueError("Qwen-Image-Edit uses sample target sizes; configure data.train.target_area instead of training.dmd.generation_shapes.")

    def latent_shape(self, batch, generation_shapes, broadcast):
        self.validate_generation_shapes(generation_shapes)
        _require_single_prompt(batch["conditioning"].get("prompt", ""))
        meta = batch.get("meta", {})
        if meta.get("target_height") is None or meta.get("target_width") is None:
            raise ValueError("Qwen-Image-Edit DMD requires meta.target_height and meta.target_width from image preprocessing or the training cache.")
        height, width = normalize_generation_shape(
            [meta["target_height"], meta["target_width"]],
            key="meta.target_height/target_width",
        )
        height, width = int(broadcast(height)), int(broadcast(width))
        return self._latent_geometry.shape(self.model, height, width)

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
