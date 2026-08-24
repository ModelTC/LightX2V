"""Shared training-cache encoding for Wan video objectives."""

from lightx2v_train.model_zoo.capability_adapters.common import (
    _training_cache_data,
    _uses_prompt_dropout,
)


def encode_wan_video_cache(model, batch, *, extra_prompts=None, conditioning_meta=None):
    """Encode one video target and all static prompt roles."""

    inputs = batch.get("inputs", {})
    if inputs.get("video") is None and inputs.get("latents") is None:
        raise ValueError("Wan cache encoding requires inputs.video or inputs.latents.")

    prompt = batch["conditioning"]["prompt"]
    prompts = {"positive": prompt, **(extra_prompts or {})}
    if _uses_prompt_dropout(model):
        prompts["unconditional"] = model.unconditional_prompt
    return _training_cache_data(
        model,
        batch,
        inputs={"latents": model.encode_to_cache_latent(batch)},
        prompts=prompts,
        contextual_roles=prompts,
        conditioning_meta=conditioning_meta,
    )
