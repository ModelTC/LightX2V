# SPDX-License-Identifier: GPL-3.0-only
# Seed-VC numerical compatibility; upstream: https://github.com/Plachtaa/seed-vc
"""Learned content projection, F0 embedding and temporal interpolation."""

import math

import torch
import torch.nn.functional as F

from .weights import linear, project


class LengthRegulator:
    def __init__(self, state, config, device):
        self.config = config
        self.state = {key: value.to(device) for key, value in state.items()}
        self.projection = linear("content_in_proj", state, device)

    def infer(self, content, frames, f0=None):
        x = project(self.projection, content).transpose(1, 2).contiguous()
        x = F.interpolate(x, size=frames, mode="nearest")
        if self.config["f0_condition"]:
            if f0 is None:
                x = x + self.state["f0_mask"].unsqueeze(-1)
            else:
                bins = self.config["n_f0_bins"]
                lower, upper = 1127 * math.log(1 + 50 / 700), 1127 * math.log(1 + 1100 / 700)
                mel = 1127 * (1 + f0 / 700).log()
                a = (bins - 2) / (upper - lower)
                b = lower * a - 1
                coarse = torch.where(mel > 0, mel * a - b, mel).round().long()
                # Preserve the release's handling of pitches outside the table.
                coarse = torch.where(coarse < 1, 1, coarse)
                coarse = torch.where(coarse < bins, coarse, 0)
                pitch = F.embedding(coarse, self.state["f0_embedding.weight"]).transpose(1, 2).contiguous()
                x = x + F.interpolate(pitch, size=frames, mode="nearest")
        count = len(self.config["sampling_ratios"])
        for index in range(count):
            p = f"model.{index * 3}"
            x = F.conv1d(x, self.state[p + ".weight"], self.state[p + ".bias"], padding=1)
            p = f"model.{index * 3 + 1}"
            x = F.group_norm(x, self.config.get("groups", 1), self.state[p + ".weight"], self.state[p + ".bias"])
            x = F.mish(x)
        p = f"model.{count * 3}"
        return F.conv1d(x, self.state[p + ".weight"], self.state[p + ".bias"]).transpose(1, 2).contiguous()
