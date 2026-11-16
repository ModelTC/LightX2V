"""One CPU reference-conditioning entry per causal runner with fixed weights."""

import hashlib
from dataclasses import dataclass

import torch


@dataclass
class ReferenceEntry:
    """CPU copies of one reference set and its global text encoding."""

    key: tuple
    prompt: str
    text: dict[str, torch.Tensor]
    device: torch.device
    latents: tuple[list[torch.Tensor], list[torch.Tensor]]
    geometry: list[tuple[int, int, int, int]]


def reference_key(references):
    # Prepared content is independent of upload paths. Waveforms already use
    # this runner's fixed sample rate, channel layout and duration clipping.
    keys = []
    for ref in references:
        if ref.kind == "image":
            keys.append((ref.kind, ref.image.mode, ref.image.size, hashlib.sha256(ref.image.tobytes()).digest()))
        elif ref.kind == "audio":
            waveform = ref.waveform.detach().cpu().contiguous()
            keys.append((ref.kind, waveform.dtype, tuple(waveform.shape), hashlib.sha256(waveform.view(torch.uint8).numpy().tobytes()).digest()))
        else:
            raise ValueError(f"Causal reference cache does not support {ref.kind!r}")
    return tuple(keys)


class MiniMaxH3ReferenceCache:
    def __init__(self):
        self.entry: ReferenceEntry | None = None

    def clear(self):
        self.entry = None

    def get(self, references, *, prompt):
        entry = self.entry
        if entry is None:
            return None
        if entry.key != reference_key(references):
            self.clear()
            return None
        # Both tasks cache the complete global encoding. A changed prompt can
        # reuse the reference latents, but needs a new global text encoding.
        text = None
        if prompt == entry.prompt:
            text = {name: value.to(entry.device, copy=True) for name, value in entry.text.items()}
        for ref, values in zip(references, entry.geometry):
            ref.num_latent_frames, ref.latent_height, ref.latent_width, ref.num_audio_latents = values
        # A request must not be able to mutate the persistent CPU tensors.
        return text, tuple([value.clone() for value in values] for values in entry.latents)

    def store(self, references, text, latents, *, prompt):
        video, audio = latents
        # Cache VAE conditions before the scheduler adds seed-dependent noise.
        cpu_latents = ([value.detach().to("cpu", copy=True) for value in video], [value.detach().to("cpu", copy=True) for value in audio])
        self.entry = ReferenceEntry(
            key=reference_key(references),
            prompt=prompt,
            text={name: value.detach().to("cpu", copy=True) for name, value in text.items()},
            device=text["prompt_embeds"].device,
            latents=cpu_latents,
            geometry=[(ref.num_latent_frames, ref.latent_height, ref.latent_width, ref.num_audio_latents) for ref in references],
        )
