"""Torch-only observation container matching OpenPI's public tensor layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Observation:
    # Images are float32 in [-1, 1], normally in BCHW layout for PyTorch.
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor | None = None
    tokenized_prompt_mask: torch.Tensor | None = None
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together")

        images: dict[str, torch.Tensor] = {}
        for key, value in data["image"].items():
            image = value
            if image.dtype == torch.uint8:
                if image.ndim != 4 or image.shape[-1] != 3:
                    raise ValueError(f"uint8 image {key!r} must use BHWC layout, got {tuple(image.shape)}")
                image = image.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
            images[key] = image

        return cls(
            images=images,
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to(self, device: torch.device | str) -> "Observation":
        def move(value):
            return None if value is None else value.to(device)

        return Observation(
            images={key: value.to(device) for key, value in self.images.items()},
            image_masks={key: value.to(device) for key, value in self.image_masks.items()},
            state=self.state.to(device),
            tokenized_prompt=move(self.tokenized_prompt),
            tokenized_prompt_mask=move(self.tokenized_prompt_mask),
            token_ar_mask=move(self.token_ar_mask),
            token_loss_mask=move(self.token_loss_mask),
        )
