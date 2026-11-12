"""Sampling PDD exports with fused output heads, without repeated backbones."""

import torch

from .core import modalities, restore


@torch.no_grad()
def sample_pdd(model, initial, condition, config, *, num_inference_steps=None):
    """Return image or joint AV latents after NFE fused PDD block evaluations.

    ``model`` must have both its trained backbone adapter and full PDD output
    heads loaded. A standard Euler sampler with only the LoRA is insufficient.
    """
    nfe = config.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
    if nfe < 1 or config.grid_size % nfe:
        raise ValueError("PDD inference steps must divide grid_size")
    block = config.grid_size // nfe
    if not config.block_size_min <= block <= config.block_size_max:
        raise ValueError("Requested PDD block size lies outside its trained range")
    grid = config.base_grid(model.device)
    state = initial
    network = model.denoiser_module()
    training = network.training
    network.eval()
    try:
        for start in range(0, config.grid_size, block):
            displacement = modalities(model.predict_pdd_block(state, start, start + block, condition, grid))
            state = restore(state, {key: value.float() + displacement[key].float() for key, value in modalities(state).items()})
    finally:
        network.train(training)
    return state
