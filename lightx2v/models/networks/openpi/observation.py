"""Torch-only observation container matching OpenPI's public tensor layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Observation:
    # Images are float32 in [-1, 1] with BCHW layout.
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None
