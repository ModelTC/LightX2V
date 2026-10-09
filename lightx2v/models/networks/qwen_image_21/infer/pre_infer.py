import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lightx2v_platform.base.global_var import AI_DEVICE


@dataclass
class TokenLayout:
    image_mask: torch.Tensor
    repeats: torch.Tensor
    rotary: torch.Tensor
    segments: list
    prefix_len: int
    target_len: int
    rotary_positions: torch.Tensor | None = None


def build_token_layout(prompt_image_mask, image_shapes, axes_dims, rope=None):
    """Expand VLM image slots and assign block-causal and three-axis positions."""
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
    rotary = torch.cat(frequencies, -1)
    rotary_positions = None
    if rope is not None:
        # Keep cos/sin in one tensor so both streams can slice the token dimension directly.
        rotary = torch.cat((rotary.real, rotary.imag), dim=-1).to(AI_DEVICE)
        rotary = rope.prepare_freqs(rotary, rotary_dim=sum(axes_dims))
        rotary_positions = rope.prepare_positions(rotary)
    else:
        rotary = rotary.to(AI_DEVICE)
    return TokenLayout(image_mask, repeats, rotary, segments, len(flags) - target_len, target_len, rotary_positions)


@dataclass
class QwenImage21PreOutput:
    hidden_states: torch.Tensor
    temb: torch.Tensor
    modulation: torch.Tensor
    layout: TokenLayout
    rotary: torch.Tensor
    rotary_positions: torch.Tensor | None = None


class QwenImage21PreInfer:
    def __init__(self):
        self.time_frequencies = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer_condition(self, weights, prompt, image_latents, layout):
        """Prepare only the immutable condition prefix, with timestep zero."""
        text = weights.txt_out.apply(F.gelu(weights.txt_in.apply(weights.txt_norm.apply(prompt)), approximate="tanh"))
        hidden = text.repeat_interleave(layout.repeats[: len(prompt)], dim=0)
        if image_latents is not None:
            hidden[layout.image_mask[: layout.prefix_len]] = weights.img_in.apply(image_latents)
        return self._prepare_output(weights, hidden, hidden.new_zeros(1), layout, layout.rotary[: layout.prefix_len])

    def infer_target(self, weights, latents, layout):
        """Prepare only the target image at the current denoising timestep."""
        hidden = weights.img_in.apply(latents)
        timestep = self.scheduler.timesteps[self.scheduler.step_index].reshape(1)
        return self._prepare_output(weights, hidden, timestep, layout, layout.rotary[layout.prefix_len :])

    def _prepare_output(self, weights, hidden, timestep, layout, rotary):
        # Match the upstream BF16 timestep round-trip before the FP32 sinusoid.
        t = timestep.to(hidden.dtype) / 1000
        if self.time_frequencies.device != t.device:
            self.time_frequencies = self.time_frequencies.to(t.device)
        angles = (1000 * t.float())[:, None] * self.time_frequencies[None]
        time = torch.cat((angles.cos(), angles.sin()), -1).to(hidden.dtype)
        temb = weights.time_out.apply(F.silu(weights.time_in.apply(time)))
        modulation = weights.modulation.apply(F.silu(temb))
        # Both streams use a sliced RoPE table, so lookup indices start at zero.
        positions = layout.rotary_positions[: len(hidden)] if layout.rotary_positions is not None else None
        return QwenImage21PreOutput(hidden, temb, modulation, layout, rotary, positions)
