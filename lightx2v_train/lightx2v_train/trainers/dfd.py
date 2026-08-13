import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v_train.runtime.distributed import (
    barrier,
    get_world_size,
    is_distributed,
    is_main_process,
    reduce_mean,
)
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .dmd.video_trainer import VideoDmdTrainer


@TRAINER_REGISTER("video_dfd")
class VideoDfdTrainer(VideoDmdTrainer):
    """Data-forcing distillation for bidirectional Wan video models.

    DFD starts each student prediction from a noised real-data latent at one
    of the finite student denoising steps. The generated x0 is then matched
    against the fake and teacher score models using the DMD objective.
    """

    trainer_name = "video_dfd"
    supports_diversity_loss = False
    supports_real_data_fake = False
    supports_ida = False

    def __init__(self, config):
        super().__init__(config)
        self.dfd_config = self.dmd_config.get(
            "dfd",
            self.dmd_config.get("data_forcing", {}),
        )
        self.dfd_real_replace_prob = float(
            self.dfd_config.get(
                "real_replace_prob",
                self.dfd_config.get(
                    "replace_prob",
                    self.dfd_config.get(
                        "prob",
                        self.dmd_config.get("dfd_real_replace_prob", 0.0),
                    ),
                ),
            )
        )
        if not 0.0 <= self.dfd_real_replace_prob <= 1.0:
            raise ValueError(
                "training.dmd.dfd.real_replace_prob must be in [0, 1], "
                f"got {self.dfd_real_replace_prob}."
            )

        self.dfd_post_train = bool(self.dfd_config.get("post_train", True))
        self.dfd_forward_kl = bool(self.dfd_config.get("forward_kl", False))
        self.dfd_use_teacher_as_fake_score = bool(
            self.dfd_config.get("use_teacher_as_fake_score", False)
        )
        self.dfd_gan_loss_weight = float(
            self.dfd_config.get("gan_loss_weight_gen", 0.0)
        )
        self.dfd_load_fake_from_generator_ckpt = bool(
            self.dfd_config.get("load_fake_from_generator_ckpt", True)
        )
        self.dfd_student_warmup_iters = int(
            self.dfd_config.get("student_warmup_iters", 0)
        )
        student_stop_iters = self.dfd_config.get("student_stop_iters")
        self.dfd_student_stop_iters = (
            int(student_stop_iters) if student_stop_iters is not None else None
        )
        if self.dfd_student_warmup_iters < 0:
            raise ValueError("training.dmd.dfd.student_warmup_iters must be >= 0.")
        if (
            self.dfd_student_stop_iters is not None
            and self.dfd_student_stop_iters < self.dfd_student_warmup_iters
        ):
            raise ValueError(
                "training.dmd.dfd.student_stop_iters must be greater than or "
                "equal to student_warmup_iters."
            )
        if self.dfd_gan_loss_weight > 0:
            logger.warning(
                "[train] video_dfd ignores dfd.gan_loss_weight_gen={} because "
                "LightX2V has no Wan discriminator module.",
                self.dfd_gan_loss_weight,
            )

    def _load_student_checkpoint(self, checkpoint_path, strict=True):
        """Prefer the consolidated state exported from an FSDP DMD checkpoint."""
        if os.path.isdir(checkpoint_path):
            consolidated_path = os.path.join(checkpoint_path, "model_state.pt")
            if os.path.isfile(consolidated_path):
                checkpoint_path = consolidated_path
        return super()._load_student_checkpoint(checkpoint_path, strict=strict)

    def setup(self, resume_ckpt_path=None):
        super().setup(resume_ckpt_path=resume_ckpt_path)
        if (
            resume_ckpt_path is None
            and self.student_checkpoint_path
            and self.dfd_load_fake_from_generator_ckpt
        ):
            self.checkpoint_manager._copy_role_model("student", "fake")
            logger.info(
                "[train] video_dfd initialized fake score from student checkpoint {}",
                self.student_checkpoint_path,
            )

        if resume_ckpt_path is None:
            student_stop = min(
                self.max_train_iters,
                self.dfd_student_stop_iters
                if self.dfd_student_stop_iters is not None
                else self.max_train_iters,
            )
            student_steps = sum(
                iteration % self.fake_update_ratio == 0
                for iteration in range(
                    self.dfd_student_warmup_iters,
                    student_stop,
                )
            )
            self.lr_scheduler = self._build_lr_scheduler(
                self.optimizer,
                num_training_steps=max(1, student_steps),
            )
            self.fake_lr_scheduler = self._build_lr_scheduler(
                self.fake_optimizer,
                num_warmup_steps=0,
                num_training_steps=max(1, self.max_train_iters),
            )

        logger.info(
            "[train] video_dfd enabled: real_replace_prob={} post_train={} "
            "forward_kl={} use_teacher_as_fake_score={} "
            "load_fake_from_generator_ckpt={} student_warmup_iters={} "
            "student_stop_iters={}",
            self.dfd_real_replace_prob,
            self.dfd_post_train,
            self.dfd_forward_kl,
            self.dfd_use_teacher_as_fake_score,
            self.dfd_load_fake_from_generator_ckpt,
            self.dfd_student_warmup_iters,
            self.dfd_student_stop_iters,
        )

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        max_train_iters = self.max_train_iters
        grad_accum_iters = max(1, int(self.gradient_accumulation_iters))
        save_every_iters = self.save_every_iters
        save_total_limit = self.save_total_limit

        logger.info(
            "[train] start method={} student_train_type={} fake_train_type={} "
            "iter={}/{} world_size={} grad_accum={} "
            "train_log_every_iters={} student_update_interval={}",
            self.training_config.get("method", self.trainer_name),
            self.student_train_type,
            self.fake_train_type,
            current_iter,
            max_train_iters,
            get_world_size(),
            grad_accum_iters,
            self.train_log_every_iters,
            self.fake_update_ratio,
        )
        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_eval)
            if current_iter == 0:
                self.run_inference(current_iter)

        samples = self._iter_train_samples()
        last_dmd = None
        while current_iter < max_train_iters:
            student_update_window = current_iter >= self.dfd_student_warmup_iters
            if self.dfd_student_stop_iters is not None:
                student_update_window = (
                    student_update_window
                    and current_iter < self.dfd_student_stop_iters
                )
            train_student = (
                student_update_window
                and current_iter % self.fake_update_ratio == 0
            )

            if train_student:
                student_result = self._train_one_stage(
                    samples,
                    stage="student",
                    grad_accum_iters=grad_accum_iters,
                )
                loss_dmd_value = student_result["dmd"]
                last_dmd = loss_dmd_value
            else:
                loss_dmd_value = last_dmd

            fake_result = self._train_one_stage(
                samples,
                stage="fake",
                grad_accum_iters=grad_accum_iters,
            )
            loss_fake_value = fake_result["loss"]

            current_iter += 1
            display_fake = reduce_mean(loss_fake_value)
            display_dmd = (
                reduce_mean(loss_dmd_value)
                if loss_dmd_value is not None
                else None
            )
            current_lr = self.lr_scheduler.get_last_lr()[0]
            if (
                current_iter == 1
                or current_iter % self.train_log_every_iters == 0
                or current_iter >= max_train_iters
            ):
                dmd_text = "nan" if display_dmd is None else f"{display_dmd:.6f}"
                logger.info(
                    "[train] iter={}/{} dfd={} fake={:.6f} lr={:.8f}",
                    current_iter,
                    max_train_iters,
                    dmd_text,
                    display_fake,
                    current_lr,
                )
                metrics = {
                    "train/fake": display_fake,
                    "train/lr": current_lr,
                }
                if display_dmd is not None:
                    metrics["train/dfd"] = display_dmd
                self.log_metrics(metrics, step=current_iter)

            if save_every_iters and current_iter % save_every_iters == 0:
                self.save_checkpoint(current_iter, save_total_limit)

            if (
                self.infer_every_iters
                and current_iter % self.infer_every_iters == 0
            ):
                self.run_inference(current_iter)

        logger.info("[train] finished iter={}/{}", current_iter, max_train_iters)

    def _train_one_stage(self, samples, stage, grad_accum_iters):
        if stage == "student":
            optimizer = self.optimizer
            scheduler = self.lr_scheduler
            params = self.trainable_params
            set_sync = self._set_student_gradient_sync
        elif stage == "fake":
            optimizer = self.fake_optimizer
            scheduler = self.fake_lr_scheduler
            params = self.fake_trainable_params
            set_sync = self._set_fake_gradient_sync
        else:
            raise ValueError(
                f"Unsupported {self.trainer_name} training stage: {stage}"
            )

        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        dmd_value = 0.0
        for micro_idx in range(grad_accum_iters):
            sample = next(samples)
            conditions = self._encode_conditions(sample)
            latent_shape = self._latent_shape(sample)
            set_sync(micro_idx == grad_accum_iters - 1)
            result = self.forward_loss(
                latent_shape,
                conditions,
                stage=stage,
                sample=sample,
            )
            loss = result["loss"]
            (loss / grad_accum_iters).backward()
            loss_value += loss.item() / grad_accum_iters
            if stage == "student":
                dmd_value += result["dmd"].item() / grad_accum_iters

        self._sync_sequence_parallel_grads(params)
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()
        if stage == "student":
            self._after_student_optimizer_step("main")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return {
            "loss": loss_value,
            **({"dmd": dmd_value} if stage == "student" else {}),
        }

    def _sample_synced_bool(self, probability):
        value = torch.rand((), device=self.model.device) < float(probability)
        if is_distributed():
            dist.broadcast(value, src=0)
        return bool(value.item())

    def _real_latent(self, sample, expected_shape):
        if sample is None:
            raise ValueError(
                "video_dfd requires the current training sample so real data "
                "can be encoded."
            )
        real_latent = self._extract_real_latents(sample).detach()
        expected_shape = tuple(int(dim) for dim in expected_shape)
        if tuple(real_latent.shape) != expected_shape:
            raise ValueError(
                "DFD real latent shape does not match the configured latent "
                f"shape: real={tuple(real_latent.shape)} expected={expected_shape}."
            )
        return real_latent

    def _sample_student_sigma(self, batch_size, device, dtype):
        step_idx = self._sample_synced_int(0, len(self.denoising_sigmas))
        sigma = self.denoising_sigmas[step_idx].to(device=device, dtype=dtype)
        return sigma.expand(int(batch_size))

    def _predict_x0(self, model, latents, sigma, condition):
        velocity = self._predict_velocity(model, latents, sigma, condition)
        expanded_sigma = self.scheduler._expand_to_ndim(sigma, latents.ndim)
        return latents - expanded_sigma * velocity

    def _predict_teacher_x0(
        self,
        latents,
        sigma,
        condition,
        negative_condition,
    ):
        if self.guidance_scale <= 1 or negative_condition is None:
            velocity = self._predict_velocity(
                self.teacher_model,
                latents,
                sigma,
                condition,
            )
        else:
            velocity = self._predict_teacher_velocity(
                latents,
                sigma,
                condition,
                negative_condition,
            )
        expanded_sigma = self.scheduler._expand_to_ndim(sigma, latents.ndim)
        return latents - expanded_sigma * velocity

    def _student_generated_x0(self, real_latent, condition, grad_enabled):
        noise = broadcast_sequence_parallel_value(
            torch.randn_like(real_latent, dtype=torch.float32)
        )
        sigma = self._sample_student_sigma(
            real_latent.shape[0],
            device=real_latent.device,
            dtype=self.running_dtype,
        )
        student_input = self.scheduler.add_noise(real_latent, noise, sigma)
        self.model.transformer.train(mode=grad_enabled)
        context = torch.enable_grad if grad_enabled else torch.no_grad
        with context():
            generated = self._predict_x0(
                self.model,
                student_input,
                sigma,
                condition,
            )
        return generated.to(dtype=self.running_dtype)

    def _teacher_score_x0(self, generated, real_latent):
        if (
            self.dfd_real_replace_prob <= 0
            or not self._sample_synced_bool(self.dfd_real_replace_prob)
        ):
            return generated

        if self.dfd_post_train:
            return generated + (real_latent - generated).detach()

        generated_index = self._sample_synced_int(0, generated.shape[0])
        real_index = self._sample_synced_int(0, real_latent.shape[0])
        teacher_score = generated.clone()
        teacher_score[generated_index] = teacher_score[generated_index] + (
            real_latent[real_index] - generated[generated_index]
        ).detach()
        return teacher_score

    def forward_loss(self, latent_shape, conditions, stage, sample=None):
        condition, negative_condition = conditions
        real_latent = self._real_latent(sample, latent_shape)
        generated = self._student_generated_x0(
            real_latent,
            condition,
            grad_enabled=stage == "student",
        )

        score_sigma = self._sample_score_sigma(
            real_latent.shape[0],
            denoised_timestep_from=None,
            denoised_timestep_to=None,
            device=self.model.device,
            dtype=self.running_dtype,
        )
        score_noise = broadcast_sequence_parallel_value(
            torch.randn_like(real_latent, dtype=torch.float32)
        )
        with torch.no_grad():
            perturbed_generated = self.scheduler.add_noise(
                generated,
                score_noise,
                score_sigma,
            )

        if stage == "fake":
            self.fake_model.transformer.train()
            velocity_fake = self._predict_velocity(
                self.fake_model,
                perturbed_generated,
                score_sigma,
                condition,
            )
            velocity_target = self.scheduler.build_train_gt(
                generated.float(),
                score_noise,
            )
            return {
                "loss": F.mse_loss(
                    velocity_fake.float(),
                    velocity_target.float(),
                    reduction="mean",
                )
            }
        if stage != "student":
            raise ValueError(
                f"Unsupported {self.trainer_name} training stage: {stage}"
            )

        with torch.no_grad():
            self.fake_model.transformer.eval()
            self.teacher_model.transformer.eval()
            teacher_score_x0 = self._teacher_score_x0(
                generated,
                real_latent,
            )
            perturbed_teacher_score = self.scheduler.add_noise(
                teacher_score_x0,
                score_noise,
                score_sigma,
            )

            if self.dfd_use_teacher_as_fake_score:
                x_pred_fake = self._predict_teacher_x0(
                    perturbed_generated,
                    score_sigma,
                    condition,
                    None,
                )
            else:
                fake_input = (
                    perturbed_teacher_score
                    if self.dfd_forward_kl
                    else perturbed_generated
                )
                x_pred_fake = self._predict_x0(
                    self.fake_model,
                    fake_input,
                    score_sigma,
                    condition,
                )

            x_pred_teacher = self._predict_teacher_x0(
                perturbed_teacher_score,
                score_sigma,
                condition,
                negative_condition,
            )

        loss_dfd = self._dmd_loss(
            generated,
            x_pred_fake,
            x_pred_teacher,
        )
        return {
            "loss": loss_dfd,
            "dmd": loss_dfd.detach(),
        }
