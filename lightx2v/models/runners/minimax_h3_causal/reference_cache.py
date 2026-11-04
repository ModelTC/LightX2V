"""One CPU reference-conditioning entry per causal runner with fixed weights."""

import hashlib

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


def reference_key(references):
    # Use prepared pixels so file replacement, resizing and reference order
    # invalidate the cache in the same way for paths and in-memory images.
    return tuple((ref.image.mode, ref.image.size, hashlib.sha256(ref.image.tobytes()).digest()) for ref in references)


class MiniMaxH3ReferenceCache:
    def __init__(self):
        self.entry = None

    def get(self, references):
        if self.entry is None:
            return None
        key, context_leaves, context_spec, latents, geometry = self.entry
        if key != reference_key(references):
            self.entry = None
            return None
        context = tree_unflatten([value.to(device) if device is not None else value for value, device in context_leaves], context_spec)
        for ref, (frames, height, width) in zip(references, geometry):
            ref.num_latent_frames, ref.latent_height, ref.latent_width = frames, height, width
        return context, latents

    def store(self, references, context, latents):
        leaves, spec = tree_flatten(context)
        # Restore only the original device tensors on a hit; masks/grids stay
        # on CPU. GPU vision features then serve all action updates in a request.
        context_leaves = [(value.detach().to("cpu", copy=True), value.device) if isinstance(value, torch.Tensor) else (value, None) for value in leaves]
        video, audio = latents
        # Cache VAE conditions before the scheduler adds seed-dependent noise.
        cpu_latents = ([value.detach().to("cpu", copy=True) for value in video], [value.detach().to("cpu", copy=True) for value in audio])
        geometry = [(ref.num_latent_frames, ref.latent_height, ref.latent_width) for ref in references]
        self.entry = reference_key(references), context_leaves, spec, cpu_latents, geometry
