# SPDX-License-Identifier: GPL-3.0-only
# Seed-VC numerical compatibility; upstream: https://github.com/Plachtaa/seed-vc
"""Forward Euler CFM, including reference-prefix masking and Seed-VC CFG."""

import torch
import torch.distributed as dist


class SeedVCScheduler:
    def sample(self, model, condition, prompt, style, steps, cfg, generator, check_stop):
        shape = (condition.shape[0], prompt.shape[1], condition.shape[1])
        x = torch.randn(shape, device=condition.device, generator=generator)
        if model.tp_size > 1:
            dist.broadcast(x, src=0, group=model.tp_group)
        prefix = prompt.shape[-1]
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prefix] = prompt
        x[..., :prefix] = 0
        timesteps = torch.linspace(0, 1, steps + 1, device=x.device)
        t = timesteps[0]
        lengths = torch.tensor([condition.shape[1]], device=x.device)
        if cfg > 0:
            prompts = torch.cat((prompt_x, torch.zeros_like(prompt_x)))
            styles = torch.cat((style, torch.zeros_like(style)))
            conditions = torch.cat((condition, torch.zeros_like(condition)))
        for index in range(steps):
            check_stop()
            if cfg > 0:
                positive, negative = model.infer(torch.cat((x, x)), prompts, lengths, t.expand(2), styles, conditions, check_stop).chunk(2)
                velocity = (1 + cfg) * positive - cfg * negative
            else:
                velocity = model.infer(x, prompt_x, lengths, t[None], style, condition, check_stop)
            dt = timesteps[index + 1] - timesteps[index]
            x = x + dt * velocity
            t = t + dt
            x[..., :prefix] = 0
        return x[..., prefix:]

    def clear(self):
        """DefaultRunner cleanup hook; sampling keeps no per-request state."""
