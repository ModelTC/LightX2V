import copy
import math
import os

import torch
from loguru import logger

from lightx2v_train.runtime.distributed import (
    barrier,
    is_main_process,
    reduce_mean,
)
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.tricks import (
    DiversityTrainerConstraints,
    RealDataFakeTrick,
)
from lightx2v_train.utils.registry import TRAINER_REGISTER

from ..dmd import DmdTrainer
from ..dmd.math import euler_step
from .checkpoint import PhasedCheckpointManager
from .config import (
    PhasedDmdConfig,
    parse_role_lora_config,
)
from .math import (
    phase_sigma,
    phased_coefficients,
    phased_forward,
    phased_velocity_target,
    raw_timesteps_to_sigmas,
    sample_score_sigma_range,
)
from .roles import PhasedRoleRegistry
from .rollout import PhasedRolloutEngine


@TRAINER_REGISTER("phased_dmd")
class PhasedDmdTrainer(DmdTrainer):
    """Dual-student phased DMD with x0-space distribution matching."""

    trainer_name = "phased_dmd"
    defer_ida_setup = True

    def __init__(self, config):
        super().__init__(config)
        self.role_registry = PhasedRoleRegistry(self)
        self.checkpoint_manager = PhasedCheckpointManager(self)
        self.rollout_engine = PhasedRolloutEngine(self)
        parsed = PhasedDmdConfig.from_mapping(
            config,
            base_config=self.parsed_dmd_schedule_config,
            dmd_config=self.dmd_config,
            infer_config=self.infer_config,
            max_train_iters=self.max_train_iters,
            default_lora_target_modules=None,
        )
        self.parsed_phased_dmd_config = parsed
        self._checkpoint_process_group = None
        self.phased_config = parsed.phased
        self.match_timestep = parsed.match_timestep
        self.match_step_index = parsed.match_step_index
        self.infer_boundary_step_index = parsed.infer_boundary_step_index
        self.score_timestep_margin = parsed.score_timestep_margin
        self.score_timestep_min = parsed.score_timestep_min
        self.score_timestep_max = parsed.score_timestep_max
        self.phased_eps = parsed.eps
        self.dmd_norm_clip_min = parsed.dmd_norm_clip_min
        self.guidance_distill = parsed.guidance_distill
        self.student_2_config = parsed.student_2
        self.fake_2_config = parsed.fake_2
        self.enable_fake_low_high = parsed.enable_fake_low_high
        self.student_2_train_type = parsed.student_2_train_type
        self.fake_2_train_type = parsed.fake_2_train_type
        self.fake_low_high_train_type = parsed.fake_low_high_train_type
        self.student_2_lora_config = parsed.student_2_lora
        self.fake_2_lora_config = parsed.fake_2_lora
        self.fake_low_high_lora_config = parsed.fake_low_high_lora
        self.student_2_optimizer_config = parsed.student_2_optimizer
        self.fake_2_optimizer_config = parsed.fake_2_optimizer
        self.fake_low_high_optimizer_config = parsed.fake_low_high_optimizer

        self.diversity_trick.validate_trainer(
            DiversityTrainerConstraints(
                supported=self.supports_diversity_loss,
                rollout_steps=self.num_inference_steps,
                high_step_count=self.match_step_index,
            ),
            self.trainer_name,
        )
        fake_real_roles = (
            ("high", "fake_real_high"),
            ("low", "fake_real_low"),
        )
        self.real_data_fake_trick = RealDataFakeTrick.from_mappings(
            {
                region: (
                    role,
                    self.phased_config.get(role, {}),
                )
                for region, role in fake_real_roles
            }
        )
        self.real_data_fake_trick.validate_schedule(
            mode="phased",
            num_train_timestep=self.num_train_timestep,
            match_timestep=self.match_timestep,
        )
        self.fake_real_high_train_type = self.fake_train_type
        self.fake_real_low_train_type = self.fake_2_train_type
        self.fake_real_high_lora_config = copy.deepcopy(self.fake_lora_config)
        self.fake_real_low_lora_config = copy.deepcopy(self.fake_2_lora_config)
        self.fake_real_high_optimizer_config = copy.deepcopy(self.fake_optimizer_config)
        self.fake_real_low_optimizer_config = copy.deepcopy(self.fake_2_optimizer_config)

    def _setup_fake_real_resources(self):
        # Phased resources are created after the Low roles are available.
        self.fake_real_model = None
        self.fake_real_optimizer = None
        self.fake_real_lr_scheduler = None
        self.fake_real_trainable_params = []
        return

    def _parse_lora_config(self, role_config, role):
        return parse_role_lora_config(
            role_config,
            role,
            self.default_lora_target_modules,
        )

    def _setup_trainable_model(self, model, role="student"):
        if role not in {
            "student_2",
            "fake_2",
            "fake_low_high",
            "fake_real_high",
            "fake_real_low",
        }:
            return super()._setup_trainable_model(model, role=role)
        return self.role_registry.setup_trainable_model(model, role)

    def _restore_trainable_model(self, model, role="student"):
        if role not in {
            "student_2",
            "fake_2",
            "fake_low_high",
            "fake_real_high",
            "fake_real_low",
        }:
            return super()._restore_trainable_model(model, role=role)
        return self.role_registry.restore_trainable_model(model, role)

    def _role_model_config(self, role):
        return self.role_registry.role_model_config(role)

    def setup(self, resume_ckpt_path=None):
        return self.role_registry.setup_resources(
            resume_ckpt_path=resume_ckpt_path,
            base_setup=super(
                PhasedDmdTrainer,
                self,
            ).setup,
        )

    def _ida_model_pairs(self):
        return self.role_registry.ida_model_pairs()

    def _region_training_steps(self):
        return (
            math.ceil(self.max_train_iters / 2),
            self.max_train_iters // 2,
        )

    def _region_for_iteration(self, current_iter):
        return "high" if current_iter % 2 == 0 else "low"

    def _teacher_branch_for_iteration(self, current_iter, region):
        if region == "high":
            return "high"
        low_iteration = current_iter // 2
        return "high" if low_iteration % 2 == 0 else "low"

    def _phase_sigma(self, dtype):
        return phase_sigma(
            match_timestep=self.match_timestep,
            device=self.student.device,
            dtype=dtype,
            warp_denoising_step=self.warp_denoising_step,
            denoising_scheduler=self.denoising_scheduler,
            num_train_timestep=self.num_train_timestep,
        )

    def _raw_timesteps_to_sigmas(self, raw_timesteps, dtype):
        return raw_timesteps_to_sigmas(
            raw_timesteps,
            device=self.student.device,
            dtype=dtype,
            warp_denoising_step=self.warp_denoising_step,
            denoising_scheduler=self.denoising_scheduler,
            num_train_timestep=self.num_train_timestep,
        )

    def _sample_score_sigma_range(
        self,
        raw_min,
        raw_max,
        device,
        dtype,
    ):
        return sample_score_sigma_range(
            raw_min,
            raw_max,
            device=device,
            dtype=dtype,
            score_timestep_min=self.score_timestep_min,
            score_timestep_max=self.score_timestep_max,
            num_train_timestep=self.num_train_timestep,
            convert_timesteps=self._raw_timesteps_to_sigmas,
            broadcast_value=broadcast_sequence_parallel_value,
        )

    def _phased_coefficients(self, sigma_s, sigma_t, ndim):
        return phased_coefficients(
            sigma_s,
            sigma_t,
            ndim,
            self.phased_eps,
        )

    def _phased_forward(self, xs, noise, sigma_s, sigma_t):
        return phased_forward(
            xs,
            noise,
            sigma_s,
            sigma_t,
            self.phased_eps,
        )

    def _phased_velocity_target(
        self,
        xs,
        noise,
        sigma_s,
        sigma_t,
    ):
        return phased_velocity_target(
            xs,
            noise,
            sigma_s,
            sigma_t,
            self.phased_eps,
        )

    def _euler_step(self, sample, velocity, sigma, sigma_next):
        return euler_step(
            sample,
            velocity,
            sigma,
            sigma_next,
        )

    def _extract_real_latents(self, sample):
        return self.rollout_engine._extract_real_latents(sample)

    def _predict_real_student_velocity(self, region, xt, sigma, condition, grad_enabled):
        return self.rollout_engine._predict_real_student_velocity(region, xt, sigma, condition, grad_enabled)

    def _predict_real_teacher_velocity(self, region, xt, sigma, condition, negative_condition):
        return self.rollout_engine._predict_real_teacher_velocity(region, xt, sigma, condition, negative_condition)

    def _diversity_step_context(self, initial_noise, conditions):
        return self.rollout_engine._diversity_step_context(initial_noise, conditions)

    def _real_data_fake_setup_context(self):
        return self.rollout_engine._real_data_fake_setup_context()

    def _real_data_fake_context(self, sample, conditions, region):
        return self.rollout_engine._real_data_fake_context(sample, conditions, region)

    def _run_region_rollout(self, condition, latent_shape, region, grad_enabled, xt=None):
        return self.rollout_engine._run_region_rollout(condition, latent_shape, region, grad_enabled, xt=xt)

    def _run_full_fake_rollout(self, condition, latent_shape, xt=None):
        return self.rollout_engine._run_full_fake_rollout(condition, latent_shape, xt=xt)

    def _teacher_velocity(self, teacher_model, xt, sigma, condition, negative_condition):
        return self.rollout_engine._teacher_velocity(teacher_model, xt, sigma, condition, negative_condition)

    def _dmd_loss_for_region(self, anchor, student_x0, sigma_s, conditions, region, teacher_branch):
        return self.rollout_engine._dmd_loss_for_region(anchor, student_x0, sigma_s, conditions, region, teacher_branch)

    def _fake_model_for_dmd(self, region, teacher_branch):
        return self.rollout_engine._fake_model_for_dmd(region, teacher_branch)

    def _fake_loss_for_score_range(self, fake_model, anchor, sigma_s, condition, raw_min, raw_max):
        return self.rollout_engine._fake_loss_for_score_range(fake_model, anchor, sigma_s, condition, raw_min, raw_max)

    def _fake_score_specifications(self, region, x_bound, x0):
        return self.rollout_engine._fake_score_specifications(region, x_bound, x0)

    def _active_student_state(self, region):
        return self.role_registry.active_student_state(region)

    def _active_fake_state(self, region):
        return self.role_registry.active_fake_state(region)

    def _active_fake_real_state(self, region):
        return self.role_registry.active_fake_real_state(region)

    def _train_student_stage(
        self,
        samples,
        grad_accum_iters,
        region,
        teacher_branch,
    ):
        optimizer, scheduler, params, set_sync = self._active_student_state(region)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_real_dmd_loss = 0.0
        total_rollout_timestep = 0.0
        total_div_loss = 0.0
        for micro_index in range(grad_accum_iters):
            sample = next(samples)
            conditions = self._encode_conditions(sample)
            latent_shape = self._latent_shape(sample)
            use_fake_real = self.real_data_fake_trick.enabled_for(region)
            set_sync(use_fake_real or micro_index == grad_accum_iters - 1)
            initial_noise = self.sample_initial_latents(latent_shape)
            (
                anchor,
                student_x0,
                sigma_s,
                rollout_timestep,
            ) = self._run_region_rollout(
                conditions[0],
                latent_shape,
                region,
                grad_enabled=True,
                xt=initial_noise,
            )
            loss = self._dmd_loss_for_region(
                anchor,
                student_x0,
                sigma_s,
                conditions,
                region,
                teacher_branch,
            )
            (loss / grad_accum_iters).backward()
            total_loss += loss.detach().item() / grad_accum_iters
            del loss, anchor, student_x0, sigma_s
            if self.diversity_trick.enabled and region == "high":
                div_raw, _ = self._backward_diversity_loss(
                    initial_noise,
                    conditions,
                    grad_accum_iters,
                )
                total_div_loss += div_raw
            del initial_noise
            if use_fake_real:
                set_sync(True)
                real_result = self.real_data_fake_trick.student_loss(
                    self._real_data_fake_context(
                        sample,
                        conditions,
                        region,
                    )
                )
                (real_result.loss / grad_accum_iters).backward()
                total_real_dmd_loss += real_result.metrics["real_dmd"].item() / grad_accum_iters
                del real_result
            total_rollout_timestep += float(rollout_timestep) / grad_accum_iters

        self._sync_sequence_parallel_grads(params)
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()
        self._after_student_optimizer_step(region)
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return (
            total_loss,
            total_real_dmd_loss,
            total_rollout_timestep,
            total_div_loss,
        )

    def _train_fake_stage(
        self,
        samples,
        grad_accum_iters,
        region,
    ):
        optimizer, scheduler, params, set_sync = self._active_fake_state(region)
        fake_model = self.fake_model if region == "high" else self.fake_2_model
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_fake_low_high_loss = 0.0
        total_fake_real_loss = 0.0
        use_fake_real = self.real_data_fake_trick.enabled_for(region)
        fake_real_state = self._active_fake_real_state(region) if use_fake_real else None
        if fake_real_state is not None:
            fake_real_optimizer = fake_real_state[0]
            fake_real_optimizer.zero_grad(set_to_none=True)
        if self.fake_low_high_optimizer is not None:
            self.fake_low_high_optimizer.zero_grad(set_to_none=True)
        for micro_index in range(grad_accum_iters):
            sample = next(samples)
            conditions = self._encode_conditions(sample)
            latent_shape = self._latent_shape(sample)
            x_bound, x0 = self._run_full_fake_rollout(
                conditions[0],
                latent_shape,
            )
            specifications = self._fake_score_specifications(
                region,
                x_bound,
                x0,
            )
            (
                _,
                active_fake_model,
                anchor,
                sigma_s,
                raw_min,
                raw_max,
            ) = specifications[0]
            if active_fake_model is not fake_model:
                raise RuntimeError("Active fake model does not match the phased score specification.")
            sync_gradients = micro_index == grad_accum_iters - 1
            set_sync(sync_gradients)
            loss = self._fake_loss_for_score_range(
                fake_model,
                anchor,
                sigma_s,
                conditions[0],
                raw_min,
                raw_max,
            )
            (loss / grad_accum_iters).backward()
            total_loss += loss.detach().item() / grad_accum_iters
            del loss
            if len(specifications) > 1:
                (
                    _,
                    fake_low_high_model,
                    low_high_anchor,
                    low_high_sigma_s,
                    low_high_raw_min,
                    low_high_raw_max,
                ) = specifications[1]
                self._set_fake_low_high_gradient_sync(sync_gradients)
                fake_low_high_loss = self._fake_loss_for_score_range(
                    fake_low_high_model,
                    low_high_anchor,
                    low_high_sigma_s,
                    conditions[0],
                    low_high_raw_min,
                    low_high_raw_max,
                )
                (fake_low_high_loss / grad_accum_iters).backward()
                total_fake_low_high_loss += fake_low_high_loss.detach().item() / grad_accum_iters
                del fake_low_high_loss
            if use_fake_real:
                fake_real_state[3](sync_gradients)
                fake_real_result = self.real_data_fake_trick.fake_loss(
                    self._real_data_fake_context(
                        sample,
                        conditions,
                        region,
                    )
                )
                (fake_real_result.loss / grad_accum_iters).backward()
                total_fake_real_loss += fake_real_result.metrics["fake_real"].item() / grad_accum_iters
                del fake_real_result
            del x_bound, x0, anchor, sigma_s, specifications

        self._sync_sequence_parallel_grads(params)
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if self.fake_low_high_optimizer is not None:
            self._sync_sequence_parallel_grads(self.fake_low_high_trainable_params)
            torch.nn.utils.clip_grad_norm_(
                self.fake_low_high_trainable_params,
                self.max_grad_norm,
            )
            self.fake_low_high_optimizer.step()
            self.fake_low_high_lr_scheduler.step()
            self.fake_low_high_optimizer.zero_grad(set_to_none=True)
        if fake_real_state is not None:
            (
                fake_real_optimizer,
                fake_real_scheduler,
                fake_real_params,
                _,
            ) = fake_real_state
            self._sync_sequence_parallel_grads(fake_real_params)
            torch.nn.utils.clip_grad_norm_(
                fake_real_params,
                self.max_grad_norm,
            )
            fake_real_optimizer.step()
            fake_real_scheduler.step()
            fake_real_optimizer.zero_grad(set_to_none=True)
        return (
            total_loss,
            total_fake_low_high_loss,
            total_fake_real_loss,
        )

    def _set_student_2_gradient_sync(self, enabled):
        self.role_registry.set_role_gradient_sync(
            "student_2",
            enabled,
        )

    def _set_fake_2_gradient_sync(self, enabled):
        self.role_registry.set_role_gradient_sync(
            "fake_2",
            enabled,
        )

    def _set_fake_low_high_gradient_sync(self, enabled):
        if self.fake_low_high_model is not None:
            self.role_registry.set_role_gradient_sync(
                "fake_low_high",
                enabled,
            )

    def _set_gradient_sync(self, enabled):
        super()._set_gradient_sync(enabled)
        self._set_student_2_gradient_sync(enabled)
        self._set_fake_2_gradient_sync(enabled)
        self._set_fake_low_high_gradient_sync(enabled)
        for role in ("fake_real_high", "fake_real_low"):
            if getattr(self, f"{role}_model", None) is not None:
                self.role_registry.set_role_gradient_sync(
                    role,
                    enabled,
                )

    def run_inference(self, current_iter):
        super().run_inference(current_iter)
        self._restore_trainable_model(
            self.student_2_model,
            role="student_2",
        )

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        grad_accum_iters = max(
            1,
            int(self.gradient_accumulation_iters),
        )
        logger.info(
            "[train] start method={} iter={}/{} grad_accum={} fake_update_ratio={} div_loss={} div_weight={} div_teacher_steps={} div_anchor_step={}",
            self.trainer_name,
            current_iter,
            self.max_train_iters,
            grad_accum_iters,
            self.fake_update_ratio,
            self.diversity_trick.enabled,
            self.diversity_trick.config.weight,
            self.diversity_trick.config.teacher_inference_steps,
            self.diversity_trick.config.anchor_step,
        )
        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_eval)
            if current_iter == 0:
                self.run_inference(current_iter)

        samples = self._iter_train_samples()
        while current_iter < self.max_train_iters:
            region = self._region_for_iteration(current_iter)
            teacher_branch = self._teacher_branch_for_iteration(
                current_iter,
                region,
            )
            (
                dmd_loss,
                real_dmd_loss,
                rollout_timestep,
                div_loss,
            ) = self._train_student_stage(
                samples,
                grad_accum_iters,
                region,
                teacher_branch,
            )
            fake_loss = 0.0
            fake_low_high_loss = 0.0
            fake_real_loss = 0.0
            for _ in range(self.fake_update_ratio):
                (
                    current_fake_loss,
                    current_fake_low_high_loss,
                    current_fake_real_loss,
                ) = self._train_fake_stage(
                    samples,
                    grad_accum_iters,
                    region,
                )
                fake_loss += current_fake_loss / self.fake_update_ratio
                fake_low_high_loss += current_fake_low_high_loss / self.fake_update_ratio
                fake_real_loss += current_fake_real_loss / self.fake_update_ratio

            current_iter += 1
            if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= self.max_train_iters:
                active_student = self._active_student_state(region)
                active_fake = self._active_fake_state(region)
                logger.info(
                    "[train] iter={}/{} region={} teacher={} dmd={:.6f} "
                    "div={:.6f} real_dmd={:.6f} fake={:.6f} "
                    "fake_low_high={:.6f} "
                    "fake_real={:.6f} "
                    "rollout_raw_t={:.2f} lr={:.8f} "
                    "fake_lr={:.8f} fake_low_high_lr={:.8f}",
                    current_iter,
                    self.max_train_iters,
                    region,
                    teacher_branch,
                    reduce_mean(dmd_loss),
                    reduce_mean(div_loss),
                    reduce_mean(real_dmd_loss),
                    reduce_mean(fake_loss),
                    reduce_mean(fake_low_high_loss),
                    reduce_mean(fake_real_loss),
                    reduce_mean(rollout_timestep),
                    active_student[1].get_last_lr()[0],
                    active_fake[1].get_last_lr()[0],
                    (self.fake_low_high_lr_scheduler.get_last_lr()[0] if self.fake_low_high_lr_scheduler is not None else 0.0),
                )

            if self.save_every_iters and current_iter % self.save_every_iters == 0:
                self.save_checkpoint(
                    current_iter,
                    self.save_total_limit,
                )
            if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                self.run_inference(current_iter)

        logger.info(
            "[train] finished iter={}/{}",
            current_iter,
            self.max_train_iters,
        )

    def _role_train_type(self, role):
        return self.checkpoint_manager._role_train_type(role)

    def _trainable_role_models(self):
        return self.checkpoint_manager._trainable_role_models()

    def _trainable_role_states(self):
        return self.checkpoint_manager._trainable_role_states()

    def _save_model_weights(self, model, save_dir, role="student"):
        return self.checkpoint_manager._save_model_weights(model, save_dir, role=role)

    def _load_model_weights(self, model, save_dir, role="student"):
        return self.checkpoint_manager._load_model_weights(model, save_dir, role=role)

    def _role_weights_dir(self, root_dir, role):
        return self.checkpoint_manager._role_weights_dir(root_dir, role)

    def _copy_fake_low_high_from_fake_2(self):
        return self.checkpoint_manager._copy_fake_low_high_from_fake_2()

    def _fast_forward_fake_low_high_scheduler(self, iteration):
        return self.checkpoint_manager._fast_forward_fake_low_high_scheduler(iteration)

    def _validate_phased_checkpoint_metadata(self, state, state_path, resume_ckpt_path):
        return self.checkpoint_manager._validate_phased_checkpoint_metadata(state, state_path, resume_ckpt_path)

    def _load_resume_state(self, resume_ckpt_path):
        return self.checkpoint_manager._load_resume_state(resume_ckpt_path)

    def _load_single_process_state(self, resume_ckpt_path):
        return self.checkpoint_manager._load_single_process_state(resume_ckpt_path)

    def _get_checkpoint_process_group(self):
        return self.checkpoint_manager._get_checkpoint_process_group()

    @staticmethod
    def _checkpoint_role_layout(dist_state_path, required_roles):
        return PhasedCheckpointManager._checkpoint_role_layout(dist_state_path, required_roles)

    def _load_distributed_state(self, resume_ckpt_path):
        return self.checkpoint_manager._load_distributed_state(resume_ckpt_path)

    def _finalize_checkpoint(self, temporary_dir, final_dir, iteration, save_total_limit):
        return self.checkpoint_manager._finalize_checkpoint(temporary_dir, final_dir, iteration, save_total_limit)

    def save_checkpoint(self, iteration, save_total_limit):
        return self.checkpoint_manager.save_checkpoint(iteration, save_total_limit)

    def _save_distributed_state(self, save_dir, iteration):
        return self.checkpoint_manager._save_distributed_state(save_dir, iteration)
