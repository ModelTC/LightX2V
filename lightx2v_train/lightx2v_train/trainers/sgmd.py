from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from loguru import logger

from lightx2v_train.runtime.distributed import (
    barrier,
    get_world_size,
    is_main_process,
    reduce_mean,
)
from lightx2v_train.runtime.sequence_parallel import (
    broadcast_sequence_parallel_value,
)
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .dmd import DmdTrainer


def normalized_fisher_loss(
    generated: torch.Tensor,
    x_pred_fake: torch.Tensor,
    x_pred_teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the SGMD Fisher loss and its per-sample normalizer."""
    reduce_dims = tuple(range(1, generated.ndim))
    with torch.no_grad():
        normalizer = (generated.float() - x_pred_teacher.float()).abs().mean(dim=reduce_dims, keepdim=True)

    per_element = 0.5 * (x_pred_fake.float() - x_pred_teacher.float()).square() / (normalizer + 1e-8)
    per_sample = per_element.flatten(1).mean(dim=1)
    return per_sample.mean(), normalizer


def generator_fake_correction_loss(
    generated: torch.Tensor,
    x_pred_fake: torch.Tensor,
    sigma: torch.Tensor,
    expand_to_ndim,
) -> torch.Tensor:
    """Construct the stop-gradient SGMD generator fake correction."""
    expanded_sigma = expand_to_ndim(
        sigma.float(),
        x_pred_fake.ndim,
    ).clamp_min(1e-8)
    with torch.no_grad():
        grad = (x_pred_fake.float() - generated.detach().float()) / expanded_sigma
        pseudo_target = x_pred_fake.float() - grad
    return 0.5 * F.mse_loss(
        x_pred_fake.float(),
        pseudo_target,
        reduction="mean",
    )


@TRAINER_REGISTER("sgmd")
class SgmdTrainer(DmdTrainer):
    """Strict video SGMD with one student and one fake-score update per iter."""

    trainer_name = "sgmd"
    fake_correction_weight = 0.1
    supports_real_data_fake = False
    supports_ida = False

    def __init__(self, config):
        super().__init__(config)
        if self.fake_update_ratio != 1:
            raise ValueError("SGMD uses one student and one fake-score update per iteration; set training.dmd.fake_update_ratio=1.")

    def sample_end_step(self):
        end_step_idx = self._sample_synced_int(
            self.diversity_trick.minimum_dmd_step_index,
            self.scheduler.num_inference_steps,
        )
        self._last_sgmd_step = end_step_idx + 1
        return end_step_idx

    def _generator_forward(
        self,
        latent_shape,
        conditions,
        initial_noise=None,
    ):
        condition, negative_condition = conditions
        (
            generated,
            denoised_timestep_from,
            denoised_timestep_to,
        ) = self.run_back_simulation(
            condition,
            latent_shape,
            grad_enabled=True,
            xt=initial_noise,
        )
        sgmd_step = self._last_sgmd_step
        sigma = self._sample_score_sigma(
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            device=self.student.device,
            dtype=self.latent_dtype,
        )
        noise = broadcast_sequence_parallel_value(
            torch.randn(
                latent_shape,
                device=self.student.device,
                dtype=torch.float32,
            )
        )
        renoised_xt = self.scheduler.add_noise(
            generated,
            noise,
            sigma,
        )

        self.fake.set_training(False)
        velocity_fake = self._predict_velocity(
            self.fake,
            renoised_xt,
            sigma,
            condition,
        )
        with torch.no_grad():
            self.teacher.set_training(False)
            velocity_teacher = self._predict_teacher_velocity(
                renoised_xt,
                sigma,
                condition,
                negative_condition,
            )

        expanded_sigma = self.scheduler._expand_to_ndim(
            sigma,
            renoised_xt.ndim,
        )
        x_pred_fake = renoised_xt - expanded_sigma * velocity_fake
        with torch.no_grad():
            x_pred_teacher = renoised_xt - expanded_sigma * velocity_teacher
        loss_fisher, normalizer = normalized_fisher_loss(
            generated,
            x_pred_fake,
            x_pred_teacher,
        )
        loss_fake_correction = generator_fake_correction_loss(
            generated,
            x_pred_fake,
            sigma,
            self.scheduler._expand_to_ndim,
        )
        loss_generator = loss_fisher - self.fake_correction_weight * loss_fake_correction
        score_context = (
            generated.detach(),
            noise,
            sigma,
            condition,
            normalizer.detach(),
        )
        return {
            "generator": loss_generator,
            "fisher": loss_fisher.detach(),
            "fake_correction": loss_fake_correction.detach(),
            "sgmd_step": sgmd_step,
        }, score_context

    def _fake_score_forward(self, score_context):
        generated, noise, sigma, condition, normalizer = score_context
        renoised_xt = self.scheduler.add_noise(
            generated,
            noise,
            sigma,
        )
        self.fake.set_training(False)
        velocity_fake = self._predict_velocity(
            self.fake,
            renoised_xt,
            sigma,
            condition,
        )
        velocity_gt = self.scheduler.build_train_gt(
            generated.float(),
            noise,
        )
        loss_v_pred = 0.5 * F.mse_loss(
            velocity_fake.float(),
            velocity_gt.detach().float(),
            reduction="mean",
        )

        with torch.no_grad():
            expanded_sigma = self.scheduler._expand_to_ndim(
                sigma,
                renoised_xt.ndim,
            )
            x_pred_fake = renoised_xt - expanded_sigma * velocity_fake
            loss_x_pred = (0.5 * (x_pred_fake.float() - generated.float()).square() / (normalizer.float() + 1e-8)).mean()
        return {
            "score": loss_v_pred,
            "v_pred": loss_v_pred.detach(),
            "x_pred": loss_x_pred.detach(),
        }

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        max_train_iters = self.max_train_iters
        grad_accum_iters = max(
            1,
            int(self.gradient_accumulation_iters),
        )
        save_every_iters = self.save_every_iters
        save_total_limit = self.save_total_limit

        logger.info(
            "[train] start method={} student_train_type={} "
            "fake_train_type={} iter={}/{} world_size={} grad_accum={} "
            "fake_correction_weight={} div_loss_enabled={} "
            "div_loss_weight={} div_teacher_steps={} div_anchor_step={}",
            self.training_config.get("method", self.trainer_name),
            self.student_train_type,
            self.fake_train_type,
            current_iter,
            max_train_iters,
            get_world_size(),
            grad_accum_iters,
            self.fake_correction_weight,
            self.diversity_trick.enabled,
            self.diversity_trick.config.weight,
            self.diversity_trick.config.teacher_inference_steps,
            self.diversity_trick.config.anchor_step,
        )
        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_val)
            if current_iter == 0:
                self.run_inference(current_iter)

        samples = self._iter_train_samples()
        while current_iter < max_train_iters:
            self.optimizer.zero_grad(set_to_none=True)
            running_generator = 0.0
            running_fisher = 0.0
            running_fake_correction = 0.0
            running_div_loss = 0.0
            running_sgmd_step = 0.0

            for micro_idx in range(grad_accum_iters):
                sample = next(samples)
                conditions = self._encode_conditions(sample)
                latent_shape = self._latent_shape(sample)
                sync_grad = micro_idx == grad_accum_iters - 1

                self._set_student_gradient_sync(sync_grad)
                self._set_fake_gradient_sync(False)
                initial_noise = self.sample_initial_latents(latent_shape)
                generator_result, score_context = self._generator_forward(
                    latent_shape,
                    conditions,
                    initial_noise=initial_noise,
                )
                (generator_result["generator"] / grad_accum_iters).backward()
                # Generator loss differentiates through the fake score with
                # respect to its input, but fake parameters are not updated here.
                self.fake_optimizer.zero_grad(set_to_none=True)

                running_generator += generator_result["generator"].detach().item() / grad_accum_iters
                running_fisher += generator_result["fisher"].item() / grad_accum_iters
                running_fake_correction += generator_result["fake_correction"].item() / grad_accum_iters
                running_sgmd_step += generator_result["sgmd_step"] / grad_accum_iters
                del generator_result, score_context

                if self.diversity_trick.enabled:
                    div_raw, div_weighted = self._backward_diversity_loss(
                        initial_noise,
                        conditions,
                        grad_accum_iters,
                    )
                    running_generator += div_weighted
                    running_div_loss += div_raw
                del initial_noise

            self._sync_sequence_parallel_grads(
                self.trainable_params,
            )
            torch.nn.utils.clip_grad_norm_(
                self.trainable_params,
                self.max_grad_norm,
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.fake_optimizer.zero_grad(set_to_none=True)
            running_v_pred = 0.0
            running_x_pred = 0.0
            for micro_idx in range(grad_accum_iters):
                sample = next(samples)
                conditions = self._encode_conditions(sample)
                latent_shape = self._latent_shape(sample)
                sync_grad = micro_idx == grad_accum_iters - 1

                with torch.no_grad():
                    unused_generator_result, score_context = self._generator_forward(
                        latent_shape,
                        conditions,
                    )
                del unused_generator_result

                self._set_fake_gradient_sync(sync_grad)
                score_result = self._fake_score_forward(score_context)
                (score_result["score"] / grad_accum_iters).backward()
                running_v_pred += score_result["v_pred"].item() / grad_accum_iters
                running_x_pred += score_result["x_pred"].item() / grad_accum_iters

            self._sync_sequence_parallel_grads(
                self.fake_trainable_params,
            )
            torch.nn.utils.clip_grad_norm_(
                self.fake_trainable_params,
                self.max_grad_norm,
            )
            self.fake_optimizer.step()
            self.fake_lr_scheduler.step()
            self.fake_optimizer.zero_grad(set_to_none=True)

            current_iter += 1
            if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= max_train_iters:
                logger.info(
                    "[train] iter={}/{} generator={:.6f} fisher={:.6f} fake_correction={:.6f} div_loss={:.6f} sgmd_step={:.2f} v_pred={:.6f} x_pred={:.6f} lr={:.8f} fake_lr={:.8f}",
                    current_iter,
                    max_train_iters,
                    reduce_mean(running_generator),
                    reduce_mean(running_fisher),
                    reduce_mean(running_fake_correction),
                    reduce_mean(running_div_loss),
                    reduce_mean(running_sgmd_step),
                    reduce_mean(running_v_pred),
                    reduce_mean(running_x_pred),
                    self.lr_scheduler.get_last_lr()[0],
                    self.fake_lr_scheduler.get_last_lr()[0],
                )

            if save_every_iters and current_iter % save_every_iters == 0:
                self.save_checkpoint(
                    current_iter,
                    save_total_limit,
                )

            if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                self.run_inference(current_iter)

        logger.info(
            "[train] finished iter={}/{}",
            current_iter,
            max_train_iters,
        )
