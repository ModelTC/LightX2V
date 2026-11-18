"""Random streams and noise for a tensor or a joint AV latent."""

import hashlib
from contextlib import contextmanager

import torch


def stream_seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little") & ((1 << 63) - 1)


def seed_generator(*parts, device="cpu"):
    return torch.Generator(device=device).manual_seed(stream_seed(*parts))


@contextmanager
def fork_rng(device):
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        yield devices


def seed_rng(seed, devices):
    torch.random.default_generator.manual_seed(seed)
    for index in devices:
        torch.cuda.default_generators[index].manual_seed(seed)


def map_latents(latents, fn):
    if torch.is_tensor(latents):
        return fn(latents, 0)
    return type(latents)(fn(latents.video, 0), fn(latents.audio, 1), latents.shape)


def noise_like(latents, generator):
    return map_latents(latents, lambda value, _: torch.randn(value.shape, device=value.device, dtype=torch.float32, generator=generator))
