"""PDMD losses and shared operations on diffusion latents."""

import hashlib
from contextlib import contextmanager

import torch
import torch.nn.functional as F


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
        return fn(latents)
    return type(latents)(fn(latents.video), fn(latents.audio), latents.shape)


def noise_like(latents, generator):
    return map_latents(latents, lambda value: torch.randn(value.shape, device=value.device, dtype=torch.float32, generator=generator))


def pdmd_critic_loss(prediction, target, clamp_max=5.0):
    return F.mse_loss(prediction.float(), target.detach().float()).clamp(max=clamp_max)


def pdmd_loss_with_stats(latents, fake_x0, teacher_x0, *, normalizer_floor=1e-5, projection_epsilon=1e-8, clamp_max=5.0):
    """Compute the projected endpoint surrogate independently for each batch item."""
    if latents.ndim < 2 or not (latents.shape == fake_x0.shape == teacher_x0.shape):
        raise ValueError("PDMD requires equal endpoint shapes with a batch dimension.")
    dims = tuple(range(1, latents.ndim))
    with torch.no_grad():
        student = latents.float()
        teacher_residual = student - teacher_x0.float()
        fake_residual = student - fake_x0.float()
        delta = teacher_residual - fake_residual
        dot = (delta * fake_residual).sum(dim=dims, keepdim=True)
        norm_sq = fake_residual.square().sum(dim=dims, keepdim=True)
        delta = delta - dot / (norm_sq + projection_epsilon) * fake_residual
        normalizer = teacher_residual.abs().mean(dim=dims, keepdim=True).clamp_min(normalizer_floor)
        update = delta / normalizer
        nonfinite = (~torch.isfinite(update)).sum()
        update = torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
        target = student - update
    return pdmd_critic_loss(latents, target, clamp_max), normalizer.mean(), update.square().mean().sqrt(), nonfinite
