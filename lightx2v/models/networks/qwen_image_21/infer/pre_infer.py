import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TokenLayout:
    image_mask: torch.Tensor
    repeats: torch.Tensor
    rotary: torch.Tensor
    segments: list
    prefix_len: int
    target_len: int


def build_token_layout(prompt_image_mask, image_shapes, axes_dims):
    """Expand VLM image slots and assign block-causal and three-axis positions."""
    device = prompt_image_mask.device
    target_len = math.prod(image_shapes[-1])
    mask = torch.cat((prompt_image_mask, prompt_image_mask.new_ones(target_len // 4)))
    repeats = torch.where(mask, 4, 1)
    image_mask = torch.repeat_interleave(mask, repeats)
    flags = image_mask.tolist()
    positions = []
    segments = []
    cursor = position = 0
    for index, (_, height, width) in enumerate(image_shapes):
        start = flags.index(True, cursor)
        if start > cursor:
            segments.append((cursor, start, True))
            positions.extend((p, p, p) for p in range(position, position + start - cursor))
            position += start - cursor
        length = height * width
        if not all(flags[start : start + length]):
            raise ValueError("VLM image slots do not match the reference latent grid")
        positions.extend((position, h, w) for h in range(-(height - height // 2), height // 2) for w in range(-(width - width // 2), width // 2))
        if index < len(image_shapes) - 1:
            segments.append((start, start + length, False))
        cursor = start + length
        position += max(height, width)
    if cursor != len(flags) or sum(flags) != sum(math.prod(s) for s in image_shapes):
        raise ValueError("Image token layout must end with exactly one target image")
    # Upstream builds its complex RoPE lookup tables in FP32 on CPU. Preserve
    # that rounding before moving them; GPU pow/trig differs at BF16 boundaries.
    indices = torch.tensor(positions)
    frequencies = []
    for axis, dim in enumerate(axes_dims):
        inv = 1.0 / torch.pow(10000, torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        phase = torch.outer(indices[:, axis], inv)
        frequencies.append(torch.polar(torch.ones_like(phase), phase))
    return TokenLayout(image_mask, repeats, torch.cat(frequencies, -1).to(device), segments, len(flags) - target_len, target_len)


@dataclass
class QwenImage21PreOutput:
    hidden_states: torch.Tensor
    temb: torch.Tensor
    modulation: torch.Tensor
    layout: TokenLayout
    rotary: torch.Tensor
    cached: bool
    rows: torch.Tensor | slice


class QwenImage21PreInfer:
    def __init__(self):
        self.time_frequencies = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, latents, prompt, image_latents, layout, cached):
        if cached:
            hidden = weights.img_in.apply(latents)
            rotary = layout.rotary[layout.prefix_len :]
        else:
            text = weights.txt_out.apply(F.gelu(weights.txt_in.apply(weights.txt_norm.apply(prompt)), approximate="tanh"))
            hidden = torch.cat((text, text.new_zeros(layout.target_len // 4, text.shape[-1])))
            hidden = hidden.repeat_interleave(layout.repeats, dim=0)
            image = latents if image_latents is None else torch.cat((image_latents, latents))
            hidden[layout.image_mask] = weights.img_in.apply(image)
            rotary = layout.rotary
        # Match the upstream BF16 timestep round-trip before the FP32 sinusoid.
        t = self.scheduler.timesteps[self.scheduler.step_index].reshape(1).to(hidden.dtype) / 1000
        t = torch.cat((t, t.new_zeros(1)))
        frequencies = self.time_frequencies.to(t.device)
        angles = (1000 * t.float())[:, None] * frequencies[None]
        time = torch.cat((angles.cos(), angles.sin()), -1).to(hidden.dtype)
        temb = weights.time_out.apply(F.silu(weights.time_in.apply(time)))
        modulation = weights.modulation.apply(F.silu(temb))
        if cached:
            # All remaining tokens use the target timestep; broadcast one row.
            rows = slice(0, 1)
        else:
            rows = (torch.arange(len(hidden), device=hidden.device) < layout.prefix_len).long()
        return QwenImage21PreOutput(hidden, temb, modulation, layout, rotary, cached, rows)
