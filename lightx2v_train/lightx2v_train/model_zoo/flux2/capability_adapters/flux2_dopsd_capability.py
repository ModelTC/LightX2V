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
from lightx2v_train.model_zoo.capability_adapters.common import _cached_condition, _training_cache_data, _uses_prompt_dropout


@dataclass(frozen=True)
class Flux2DopsdReference:
    latents: torch.Tensor
    ids: torch.Tensor


class Flux2DopsdCapability(BoundCapability, DopsdCapability):
    """Flux2 tensor preparation and denoising required by DOPSD."""

    @property
    def device(self) -> torch.device:
        return self.model.device

    def encode_training_cache(self, batch):
        from lightx2v_train.trainers.dopsd_core import DopsdConfig

        image = batch.get("inputs", {}).get("target_pixel_values")
        if image is None:
            raise ValueError("DOPSD cache construction requires inputs.target_pixel_values.")
        image = self._image(image, self.model.running_dtype)
        latent = self.model.vae.encode(image).latent_dist.mode()
        prompt = batch["conditioning"]["prompt"]
        config = DopsdConfig.from_training_config(self.model.config["training"])
        teacher_prompts = config.teacher_prompts(prompt)
        prompts = {
            "positive": prompt,
            "teacher": teacher_prompts,
        }
        contextual_roles = {"positive"}
        if _uses_prompt_dropout(self.model):
            prompts["unconditional"] = self.model.unconditional_prompt
            prompts["teacher_unconditional"] = config.teacher_prompts(self.model.unconditional_prompt)
            contextual_roles.add("unconditional")
        return _training_cache_data(
            self.model,
            batch,
            inputs={"dopsd_reference_latents": self.model._normalize_patch_latents(latent)},
            prompts=prompts,
            contextual_roles=contextual_roles,
            conditioning_meta={"teacher_prompt": teacher_prompts},
        )

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

    def _cached_reference(self, latent: torch.Tensor) -> Flux2DopsdReference:
        reference_ids = Flux2KleinPipeline._prepare_image_ids([latent[:1]])
        reference_ids = reference_ids.repeat(latent.shape[0], 1, 1).to(self.device)
        return Flux2DopsdReference(
            latents=Flux2KleinPipeline._pack_latents(latent),
            ids=reference_ids,
        )

    @staticmethod
    def _dimension(meta, key):
        value = meta[key]
        if torch.is_tensor(value):
            value = value.reshape(-1)[0].item()
        elif isinstance(value, (list, tuple)):
            value = value[0]
        return int(value)

    def prepare_training_batch(
        self,
        batch: Mapping[str, Any],
        teacher_prompts: list[str],
        running_dtype: torch.dtype,
    ) -> DopsdPreparedBatch:
        cached_latent = batch.get("inputs", {}).get("dopsd_reference_latents")
        if cached_latent is not None:
            cached_teacher_prompt = batch["conditioning"].get("teacher_prompt")
            if list(cached_teacher_prompt or []) != list(teacher_prompts):
                raise ValueError("Cached DOPSD teacher prompt does not match the current training configuration. Rebuild the cache.")
            cached_latent = cached_latent.to(device=self.device, dtype=running_dtype)
            meta = batch.get("meta", {})
            height = self._dimension(meta, "target_height")
            width = self._dimension(meta, "target_width")
            initial_state, state_ids, latent_hw = self._initial_state(height, width, generator=None)
            return DopsdPreparedBatch(
                initial_state=initial_state,
                state_ids=state_ids,
                student_condition=_cached_condition(batch, self.model),
                teacher_condition=_cached_condition(
                    batch,
                    self.model,
                    role="teacher_unconditional" if batch["conditioning"].get("active") == "unconditional" else "teacher",
                ),
                teacher_reference=self._cached_reference(cached_latent),
                latent_hw=latent_hw,
            )

        cache_path = batch.get("meta", {}).get("training_cache_path")
        if cache_path is not None:
            raise KeyError(f"Training cache {cache_path} has no inputs.dopsd_reference_latents entry.")

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
