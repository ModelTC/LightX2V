"""DOPSD capability for Flux2 models."""

import torch
import torch.nn.functional as F

from lightx2v_train.model_capabilities import (
    BoundCapability,
    DopsdCapability,
    DopsdLossResult,
    DopsdStepContext,
)


class Flux2DopsdCapability(BoundCapability, DopsdCapability):
    @property
    def device(self):
        return self.model.device

    def configure_adapters(
        self,
        rank,
        alpha,
        target_modules,
        student_adapter,
        teacher_adapter,
        initialize_teacher,
    ) -> None:
        self.model.add_dual_lora(
            rank,
            alpha,
            target_modules,
            student_adapter=student_adapter,
            teacher_adapter=teacher_adapter,
            init_teacher_from_student=initialize_teacher,
        )
        self.set_training(student_adapter, teacher_adapter)

    def parameters(self):
        return self.model.trainable_parameters()

    def compute_loss(
        self,
        batch,
        context: DopsdStepContext,
    ) -> DopsdLossResult:
        image = batch["inputs"].get("target_image")
        if image is None:
            raise ValueError("D-OPSD requires inputs.target_image.")
        image = image.to(
            device=self.device,
            dtype=context.running_dtype,
        )
        if image.shape[0] != 1:
            raise ValueError("D-OPSD only supports physical batch size 1.")
        _, _, height, width = image.shape
        latent_hw = (height // 16, width // 16)
        timestep_scale = float(context.scheduler.num_train_timesteps)

        with torch.no_grad():
            student_condition = self.model.encode_condition(batch)
            teacher_condition = self.model.encode_prompt_text(context.teacher_prompts(batch["conditioning"]["prompt"]))
            reference_latents, reference_ids = self.model.prepare_reference_image_latents(image)
            initial_latents, latent_ids = self.model.prepare_dopsd_initial_latents(
                height,
                width,
            )
            context.scheduler.set_timesteps(
                context.num_training_steps,
                latent_hw=latent_hw,
            )
            timesteps = context.scheduler.infer_timesteps

        latents = initial_latents
        total_loss = 0.0
        weights = context.step_loss_weights(len(timesteps))
        student_trajectory = []
        teacher_trajectory = []
        for step_index, timestep in enumerate(timesteps):
            time = (timestep.reshape(1) / timestep_scale).to(
                device=self.device,
                dtype=context.running_dtype,
            )
            if step_index + 1 < len(timesteps):
                next_time = timesteps[step_index + 1].reshape(1)
                next_time = next_time / timestep_scale
            else:
                next_time = torch.zeros_like(time)
            next_time = next_time.to(
                device=self.device,
                dtype=context.running_dtype,
            )
            latents = latents.detach().requires_grad_(True)
            with torch.no_grad():
                teacher_velocity = self.model.predict_velocity(
                    latents,
                    time,
                    teacher_condition,
                    latent_ids,
                    context.teacher_adapter,
                    teacher_image_latents=reference_latents,
                    teacher_image_latent_ids=reference_ids,
                )
                teacher_x0 = latents - time.reshape(1, 1, 1) * teacher_velocity
            student_velocity = self.model.predict_velocity(
                latents,
                time,
                student_condition,
                latent_ids,
                context.student_adapter,
            )
            student_x0 = latents - time.reshape(1, 1, 1) * student_velocity
            latents = latents + student_velocity * (next_time - time).reshape(1, 1, 1)
            total_loss = total_loss + weights[step_index] * F.mse_loss(
                student_x0,
                teacher_x0.detach(),
            )
            if context.collect_trajectory:
                student_trajectory.append(student_x0.detach())
                teacher_trajectory.append(teacher_x0.detach())

        return DopsdLossResult(
            loss=total_loss / sum(weights),
            student_trajectory=tuple(student_trajectory),
            teacher_trajectory=tuple(teacher_trajectory),
            latent_ids=latent_ids if context.collect_trajectory else None,
            height=height if context.collect_trajectory else None,
            width=width if context.collect_trajectory else None,
        )

    def ema_update(self, student_adapter, teacher_adapter, decay) -> None:
        self.model.ema_update_lora_adapter(
            student_adapter,
            teacher_adapter,
            decay,
        )

    def decode_trajectory(self, trajectory, latent_ids):
        return self.model.decode_packed_x0_to_images(trajectory, latent_ids)

    def set_training(self, student_adapter, teacher_adapter) -> None:
        self.model.set_dual_lora_trainable(
            student_adapter,
            teacher_adapter,
        )

    def set_eval(self) -> None:
        self.model.set_denoiser_eval()

    def set_active_adapter(self, adapter_name) -> None:
        self.model.set_active_adapter(adapter_name)

    def encode_prompt(self, prompts):
        return self.model.encode_prompt_text(prompts)

    def prepare_reference(self, image):
        return self.model.prepare_reference_image_latents(image)

    def initial_latents(self, height, width, generator=None):
        return self.model.prepare_dopsd_initial_latents(
            height,
            width,
            generator=generator,
        )

    def predict_velocity(
        self,
        latents,
        time,
        condition,
        latent_ids,
        adapter_name,
        **kwargs,
    ):
        return self.model.predict_velocity(
            latents,
            time,
            condition,
            latent_ids,
            adapter_name,
            **kwargs,
        )

    def load_adapter(self, path, adapter_name, weights_subdir=None) -> None:
        self.model.load_lora_weights_for_resume(
            path,
            adapter_name=adapter_name,
            weights_subdir=weights_subdir,
        )

    def save_adapter(self, path, adapter_name, weights_subdir=None) -> None:
        self.model.save_lora_weights(
            path,
            adapter_name=adapter_name,
            weights_subdir=weights_subdir,
        )

    def copy_adapter(self, source_adapter, target_adapter) -> None:
        self.model.copy_lora_adapter_weights(
            source_adapter,
            target_adapter,
        )
