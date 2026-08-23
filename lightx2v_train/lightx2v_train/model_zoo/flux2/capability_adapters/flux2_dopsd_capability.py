from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from diffusers import Flux2KleinPipeline

from lightx2v_train.model_capabilities import (
    BoundCapability,
    DopsdCapability,
    DopsdPreparedBatch,
    DopsdPreparedTeacherBatch,
)


@dataclass(frozen=True)
class Flux2DopsdReference:
    latents: torch.Tensor
    ids: torch.Tensor


class Flux2DopsdCapability(BoundCapability, DopsdCapability):
    """Flux2 tensor preparation and denoising required by DOPSD."""

    @property
    def device(self) -> torch.device:
        return self.model.device

    def _image(self, image: torch.Tensor, running_dtype: torch.dtype) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"DOPSD requires one reference image per rank with shape [1, C, H, W], got {tuple(image.shape)}.")
        return image.to(device=self.device, dtype=running_dtype)

    def _initial_state(
        self,
        height: int,
        width: int,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        unpacked_state = self.model.prepare_infer_latents(height, width, generator=generator)
        state_ids = Flux2KleinPipeline._prepare_latent_ids(unpacked_state).to(self.device)
        state = Flux2KleinPipeline._pack_latents(unpacked_state)
        return state, state_ids, tuple(unpacked_state.shape[-2:])

    def _reference(self, image: torch.Tensor) -> Flux2DopsdReference:
        encoded = self.model.vae.encode(image).latent_dist.mode()
        normalized = self.model._normalize_patch_latents(encoded)
        reference_ids = Flux2KleinPipeline._prepare_image_ids([normalized[:1]])
        reference_ids = reference_ids.repeat(normalized.shape[0], 1, 1).to(self.device)
        reference_latents = Flux2KleinPipeline._pack_latents(normalized)
        return Flux2DopsdReference(
            latents=reference_latents.to(device=self.device, dtype=self.model.running_dtype),
            ids=reference_ids,
        )

    def prepare_training_batch(
        self,
        batch: Mapping[str, Any],
        teacher_prompts: list[str],
        running_dtype: torch.dtype,
    ) -> DopsdPreparedBatch:
        image = batch.get("inputs", {}).get("target_pixel_values")
        if image is None:
            raise ValueError("DOPSD requires inputs.target_pixel_values.")
        image = self._image(image, running_dtype)
        height, width = image.shape[-2:]
        initial_state, state_ids, latent_hw = self._initial_state(height, width, generator=None)
        return DopsdPreparedBatch(
            initial_state=initial_state,
            state_ids=state_ids,
            student_condition=self.model.encode_condition(batch),
            teacher_condition=self.model.encode_prompt_condition(teacher_prompts),
            teacher_reference=self._reference(image),
            latent_hw=latent_hw,
        )

    def prepare_teacher_batch(
        self,
        reference_image: torch.Tensor,
        teacher_prompts: list[str],
        running_dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> DopsdPreparedTeacherBatch:
        image = self._image(reference_image, running_dtype)
        height, width = image.shape[-2:]
        initial_state, state_ids, latent_hw = self._initial_state(height, width, generator)
        return DopsdPreparedTeacherBatch(
            initial_state=initial_state,
            state_ids=state_ids,
            condition=self.model.encode_prompt_condition(teacher_prompts),
            reference=self._reference(image),
            latent_hw=latent_hw,
            height=height,
            width=width,
        )

    def predict_velocity(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: Any,
        state_ids: torch.Tensor,
        reference: Any = None,
    ) -> torch.Tensor:
        hidden_states = state
        image_ids = state_ids
        if reference is not None:
            if not isinstance(reference, Flux2DopsdReference):
                raise TypeError(f"Flux2 DOPSD expected Flux2DopsdReference, got {type(reference).__name__}.")
            hidden_states = torch.cat([state, reference.latents], dim=1)
            image_ids = torch.cat([state_ids, reference.ids], dim=1)

        velocity = self.model.transformer(
            hidden_states=hidden_states,
            timestep=time,
            guidance=None,
            encoder_hidden_states=condition["prompt_embed"],
            txt_ids=condition["text_ids"],
            img_ids=image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        return velocity[:, : state.shape[1]]

    @torch.no_grad()
    def decode_state(self, state: torch.Tensor, state_ids: torch.Tensor):
        unpacked = Flux2KleinPipeline._unpack_latents_with_ids(state, state_ids)
        return self.model.decode_latent(unpacked)
