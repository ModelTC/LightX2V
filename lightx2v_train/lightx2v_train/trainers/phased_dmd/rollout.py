from functools import partial

import torch
import torch.nn.functional as F

from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.tricks import (
    DiversityStepContext,
    RealDataFakeSetupContext,
    RealDataFakeStepContext,
)


class PhasedRolloutEngine:
    """Run phased student and fake rollouts over trainer-owned models."""

    def __init__(self, owner):
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def __setattr__(self, name, value):
        setattr(self.owner, name, value)

    def _run_region_rollout(
        self,
        condition,
        latent_shape,
        region,
        grad_enabled,
        xt=None,
    ):
        self.scheduler.set_timesteps(
            self.num_inference_steps,
            sigmas=[float(sigma) for sigma in self.denoising_sigmas.detach().cpu()],
            latent_hw=latent_shape[-2:],
            device=self.student.device,
        )
        if region == "high":
            min_step_index = self.diversity_trick.minimum_dmd_step_index if grad_enabled else 0
            gradient_step_index = self._sample_synced_int(
                min_step_index,
                self.match_step_index,
            )
        else:
            gradient_step_index = self._sample_synced_int(
                self.match_step_index,
                self.num_inference_steps,
            )

        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        self.student.set_training(True)
        self.student_2.set_training(True)
        for step_index in range(gradient_step_index + 1):
            active_model = self.student if step_index < self.match_step_index else self.student_2
            sigma = self.scheduler.sigma_at(
                step_index,
                device=self.student.device,
                dtype=self.latent_dtype,
            )
            keep_gradient = grad_enabled and step_index == gradient_step_index
            context = torch.enable_grad if keep_gradient else torch.no_grad
            with context():
                velocity = self._predict_velocity(
                    active_model,
                    xt,
                    sigma,
                    condition,
                )
            if step_index < gradient_step_index:
                xt, _ = self.scheduler.step_by_index(
                    velocity,
                    step_index,
                    xt,
                )

        zero_sigma = torch.zeros_like(sigma)
        student_x0 = self._euler_step(
            xt,
            velocity,
            sigma,
            zero_sigma,
        )
        if region == "high":
            sigma_s = self._phase_sigma(
                dtype=self.latent_dtype,
            )
            anchor = self._euler_step(
                xt,
                velocity,
                sigma,
                sigma_s,
            )
        else:
            sigma_s = zero_sigma
            anchor = student_x0
        raw_gradient_timestep = self._raw_timestep_from_warped_step(self.denoising_steps[gradient_step_index])
        return (
            anchor.to(dtype=self.latent_dtype),
            student_x0.to(dtype=self.latent_dtype),
            sigma_s,
            raw_gradient_timestep,
        )

    @torch.no_grad()
    def _run_full_fake_rollout(
        self,
        condition,
        latent_shape,
        xt=None,
    ):
        self.scheduler.set_timesteps(
            self.num_inference_steps,
            sigmas=[float(sigma) for sigma in self.denoising_sigmas.detach().cpu()],
            latent_hw=latent_shape[-2:],
            device=self.student.device,
        )
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        self.student.set_training(False)
        self.student_2.set_training(False)
        x_bound = None
        for step_index in range(self.num_inference_steps):
            active_model = self.student if step_index < self.match_step_index else self.student_2
            sigma = self.scheduler.sigma_at(
                step_index,
                device=self.student.device,
                dtype=self.latent_dtype,
            )
            velocity = self._predict_velocity(
                active_model,
                xt,
                sigma,
                condition,
            )
            xt, _ = self.scheduler.step_by_index(
                velocity,
                step_index,
                xt,
            )
            if step_index + 1 == self.match_step_index:
                x_bound = xt.detach()
        if x_bound is None:
            raise RuntimeError("Full phased rollout did not reach the phase boundary.")
        return (
            x_bound.to(dtype=self.latent_dtype),
            xt.detach().to(dtype=self.latent_dtype),
        )

    def _teacher_velocity(
        self,
        teacher_model,
        xt,
        sigma,
        condition,
        negative_condition,
    ):
        velocity_cond = self._predict_velocity(
            teacher_model,
            xt,
            sigma,
            condition,
        )
        if negative_condition is None:
            return velocity_cond
        velocity_uncond = self._predict_velocity(
            teacher_model,
            xt,
            sigma,
            negative_condition,
        )
        return self._do_cfg(
            velocity_cond,
            velocity_uncond,
            self.guidance_distill,
            self.cfg_norm,
        )

    def _dmd_loss_for_region(
        self,
        anchor,
        student_x0,
        sigma_s,
        conditions,
        region,
        teacher_branch,
    ):
        condition, negative_condition = conditions
        if teacher_branch == "high":
            raw_min = self.match_timestep + self.score_timestep_margin
            raw_max = self.num_train_timestep
            teacher_model = self.teacher
        else:
            raw_min = 1
            raw_max = self.match_timestep
            teacher_model = self.teacher_2
        sigma_t = self._sample_score_sigma_range(
            raw_min,
            raw_max,
            anchor.device,
            self.latent_dtype,
        )
        noise = broadcast_sequence_parallel_value(torch.randn_like(anchor, dtype=torch.float32))
        fake_model = self._fake_model_for_dmd(
            region,
            teacher_branch,
        )
        with torch.no_grad():
            score_xt = self._phased_forward(
                anchor.detach(),
                noise,
                sigma_s,
                sigma_t,
            )
            fake_model.set_training(False)
            teacher_model.set_training(False)
            velocity_fake = self._predict_velocity(
                fake_model,
                score_xt,
                sigma_t,
                condition,
            )
            velocity_teacher = self._teacher_velocity(
                teacher_model,
                score_xt,
                sigma_t,
                condition,
                negative_condition,
            )
            zero_sigma = torch.zeros_like(sigma_t)
            fake_x0 = self._euler_step(
                score_xt,
                velocity_fake,
                sigma_t,
                zero_sigma,
            )
            teacher_x0 = self._euler_step(
                score_xt,
                velocity_teacher,
                sigma_t,
                zero_sigma,
            )
        return self._dmd_loss(
            student_x0,
            fake_x0,
            teacher_x0,
            norm_clip_min=self.dmd_norm_clip_min,
        )

    def _fake_model_for_dmd(self, region, teacher_branch):
        if region == "high":
            return self.fake
        if teacher_branch == "high" and self.fake_low_high_model is not None:
            return self.fake_low_high
        return self.fake_2

    def _fake_loss_for_score_range(
        self,
        fake_model,
        anchor,
        sigma_s,
        condition,
        raw_min,
        raw_max,
    ):
        sigma_t = self._sample_score_sigma_range(
            raw_min,
            raw_max,
            anchor.device,
            self.latent_dtype,
        )
        noise = broadcast_sequence_parallel_value(torch.randn_like(anchor, dtype=torch.float32))
        with torch.no_grad():
            score_xt = self._phased_forward(
                anchor.detach(),
                noise,
                sigma_s,
                sigma_t,
            )
            velocity_target = self._phased_velocity_target(
                anchor.detach(),
                noise,
                sigma_s,
                sigma_t,
            )
        fake_model.set_training(True)
        velocity_fake = self._predict_velocity(
            fake_model,
            score_xt,
            sigma_t,
            condition,
        )
        return F.mse_loss(
            velocity_fake.float(),
            velocity_target.float(),
            reduction="mean",
        )

    def _fake_score_specifications(self, region, x_bound, x0):
        if region == "high":
            anchor = x_bound
            sigma_s = self._phase_sigma(
                dtype=self.latent_dtype,
            )
            fake_model = self.fake
            raw_min = self.match_timestep + self.score_timestep_margin
            raw_max = self.num_train_timestep
        else:
            anchor = x0
            sigma_s = torch.zeros(
                1,
                device=x0.device,
                dtype=self.latent_dtype,
            )
            fake_model = self.fake_2
            raw_min = 1
            raw_max = self.num_train_timestep if self.fake_low_high_model is None else self.match_timestep
        specifications = [
            (
                "fake",
                fake_model,
                anchor,
                sigma_s,
                raw_min,
                raw_max,
            )
        ]
        if self.fake_low_high_model is not None:
            specifications.append(
                (
                    "fake_low_high",
                    self.fake_low_high,
                    x0,
                    torch.zeros(
                        1,
                        device=x0.device,
                        dtype=self.latent_dtype,
                    ),
                    (self.match_timestep + self.score_timestep_margin),
                    self.num_train_timestep,
                )
            )
        return tuple(specifications)

    def _extract_real_latents(self, sample):
        return self.student.extract_real_latents(
            sample,
            self.latent_dtype,
            broadcast_sequence_parallel_value,
        )

    def _predict_real_student_velocity(
        self,
        region,
        xt,
        sigma,
        condition,
        grad_enabled,
    ):
        context = torch.enable_grad if grad_enabled else torch.no_grad
        with context():
            student_model = self.student if region == "high" else self.student_2
            student_model.set_training(grad_enabled)
            velocity = self._predict_velocity(
                student_model,
                xt,
                sigma,
                condition,
            )
        return velocity

    def _predict_real_teacher_velocity(
        self,
        region,
        xt,
        sigma,
        condition,
        negative_condition,
    ):
        teacher_model = self.teacher if region == "high" else self.teacher_2
        teacher_model.set_training(False)
        return self._teacher_velocity(
            teacher_model,
            xt,
            sigma,
            condition,
            negative_condition,
        )

    def _diversity_step_context(self, initial_noise, conditions):
        return DiversityStepContext(
            initial_noise=initial_noise.detach(),
            condition=conditions[0],
            negative_condition=conditions[1],
            latent_hw=initial_noise.shape[-2:],
            device=self.student.device,
            dtype=self.latent_dtype,
            predict_teacher_velocity=partial(
                self._teacher_velocity,
                self.teacher,
            ),
            predict_student_velocity=partial(
                self._predict_velocity,
                self.student,
            ),
            student_scheduler=self.scheduler,
            expand_to_ndim=self.scheduler._expand_to_ndim,
        )

    def _real_data_fake_setup_context(self):
        return RealDataFakeSetupContext(
            mode="phased",
            scheduler=self.scheduler,
            denoising_scheduler=self.denoising_scheduler,
            num_train_timestep=self.num_train_timestep,
            warp_denoising_step=self.warp_denoising_step,
            match_timestep=self.match_timestep,
            score_timestep_margin=self.score_timestep_margin,
            score_timestep_min=self.score_timestep_min,
            score_timestep_max=self.score_timestep_max,
            phased_eps=self.phased_eps,
        )

    def _real_data_fake_context(self, sample, conditions, region):
        return RealDataFakeStepContext(
            region=region,
            fake_model=(self.fake_real_high if region == "high" else self.fake_real_low),
            sample=sample,
            condition=conditions[0],
            negative_condition=conditions[1],
            device=self.student.device,
            dtype=self.latent_dtype,
            extract_real_latents=self._extract_real_latents,
            sample_synced_int=self._sample_synced_int,
            broadcast_noise=broadcast_sequence_parallel_value,
            predict_student_velocity=self._predict_real_student_velocity,
            predict_fake_velocity=self._predict_velocity,
            predict_teacher_velocity=(self._predict_real_teacher_velocity),
            dmd_loss=(
                lambda student_x0, fake_x0, teacher_x0: self._dmd_loss(
                    student_x0,
                    fake_x0,
                    teacher_x0,
                    norm_clip_min=self.dmd_norm_clip_min,
                )
            ),
        )
