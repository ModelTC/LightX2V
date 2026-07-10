import os

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from loguru import logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from lightx2v_train.runtime.distributed import barrier, get_world_size, is_main_process, reduce_mean
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .dmd import VideoDmdTrainer


@TRAINER_REGISTER("video_dfd")
class VideoDfdTrainer(VideoDmdTrainer):
    """Data Forcing Distillation trainer following the reference DMD2/DFD update.

    Unlike `VideoDmdTrainer`, the student is trained from real-data noised
    states sampled on the finite distillation step list. This mirrors the
    reference DFD post-training path where `input_student = q_t(real)`.
    """

    trainer_name = "video_dfd"

    def __init__(self, config):
        super().__init__(config)
        self.dfd_post_train = bool(self.dfd_config.get("post_train", True))
        self.dfd_forward_kl = bool(self.dfd_config.get("forward_kl", False))
        self.dfd_use_teacher_as_fake_score = bool(self.dfd_config.get("use_teacher_as_fake_score", False))
        self.dfd_gan_loss_weight = float(self.dfd_config.get("gan_loss_weight_gen", 0.0))
        self.dfd_load_fake_from_generator_ckpt = bool(self.dfd_config.get("load_fake_from_generator_ckpt", True))
        self.dfd_student_warmup_iters = int(self.dfd_config.get("student_warmup_iters", 0))
        student_stop_iters = self.dfd_config.get("student_stop_iters")
        self.dfd_student_stop_iters = int(student_stop_iters) if student_stop_iters is not None else None
        if self.dfd_gan_loss_weight > 0:
            logger.warning(
                "[train] video_dfd ignores dfd.gan_loss_weight_gen={} because LightX2V has no Wan discriminator module.",
                self.dfd_gan_loss_weight,
            )

    def setup(self, resume_ckpt_path=None):
        super().setup(resume_ckpt_path=resume_ckpt_path)
        if resume_ckpt_path is None and self.dfd_load_fake_from_generator_ckpt:
            self._load_fake_score_checkpoint(self.student_checkpoint_path)
        logger.info(
            "[train] video_dfd original update enabled: post_train={} forward_kl={} use_teacher_as_fake_score={} load_fake_from_generator_ckpt={} student_warmup_iters={} student_stop_iters={}",
            self.dfd_post_train,
            self.dfd_forward_kl,
            self.dfd_use_teacher_as_fake_score,
            self.dfd_load_fake_from_generator_ckpt,
            self.dfd_student_warmup_iters,
            self.dfd_student_stop_iters,
        )

    def _load_fake_score_checkpoint(self, checkpoint_path):
        if not checkpoint_path:
            return
        checkpoint_path = os.fspath(checkpoint_path)
        dist_state_path = os.path.join(checkpoint_path, "dist_state") if os.path.isdir(checkpoint_path) else None
        if dist_state_path is None or not os.path.isdir(dist_state_path):
            logger.warning("[train] video_dfd cannot warm-start fake score; dist_state/ not found in {}", checkpoint_path)
            return
        if not self.fake_model.is_fsdp2_wrapped():
            logger.warning("[train] video_dfd fake-score warm-start from dist_state requires FSDP2-wrapped fake model.")
            return

        options = StateDictOptions(ignore_frozen_params=True, strict=False)
        fake_model_state, fake_optim_state = get_state_dict(self.fake_model.fsdp2_state_module(), self.fake_optimizer, options=options)
        state = {
            "fake_model": fake_model_state,
            "fake_optimizer": fake_optim_state,
        }
        dcp.load(state, checkpoint_id=dist_state_path)
        set_state_dict(
            self.fake_model.fsdp2_state_module(),
            self.fake_optimizer,
            model_state_dict=state["fake_model"],
            optim_state_dict=state["fake_optimizer"],
            options=options,
        )
        logger.info("[train] video_dfd loaded fake-score checkpoint from {}", dist_state_path)

    def train(self):
        if self.dfd_student_warmup_iters <= 0:
            return super().train()

        original_ratio = self.fake_update_ratio
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if current_iter < self.dfd_student_warmup_iters:
            logger.info("[train] video_dfd delaying student updates until iter={}", self.dfd_student_warmup_iters)
        self.fake_update_ratio = original_ratio
        return self._train_with_student_warmup(current_iter)

    def _train_with_student_warmup(self, current_iter):
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        max_train_iters = self.max_train_iters
        grad_accum_iters = max(1, int(self.gradient_accumulation_iters))
        save_every_iters = self.save_every_iters
        save_total_limit = self.save_total_limit

        logger.info(
            "[train] start method={} train_type={} iter={}/{} world_size={} grad_accum={} train_log_every_iters={} fake_update_ratio={} student_warmup_iters={}",
            self.training_config.get("method", self.trainer_name),
            self.train_type,
            current_iter,
            max_train_iters,
            get_world_size(),
            grad_accum_iters,
            self.train_log_every_iters,
            self.fake_update_ratio,
            self.dfd_student_warmup_iters,
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
                student_update_window = student_update_window and current_iter < self.dfd_student_stop_iters
            train_student = student_update_window and current_iter % self.fake_update_ratio == 0

            if train_student:
                loss_dmd_value = self._train_one_stage(samples, stage="student", grad_accum_iters=grad_accum_iters)
                last_dmd = loss_dmd_value
            else:
                loss_dmd_value = last_dmd

            loss_fake_value = self._train_one_stage(samples, stage="fake", grad_accum_iters=grad_accum_iters)

            current_iter += 1
            display_fake = reduce_mean(loss_fake_value)
            display_dmd = reduce_mean(loss_dmd_value) if loss_dmd_value is not None else None
            if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= max_train_iters:
                dmd_text = "nan" if display_dmd is None else f"{display_dmd:.6f}"
                logger.info(
                    "[train] iter={}/{} dfd={} fake={:.6f} lr={:.8f}",
                    current_iter,
                    max_train_iters,
                    dmd_text,
                    display_fake,
                    self.lr_scheduler.get_last_lr()[0],
                )

            if save_every_iters and current_iter % save_every_iters == 0:
                self.save_checkpoint(current_iter, save_total_limit)

            if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                self.run_inference(current_iter)

        logger.info("[train] finished iter={}/{}", current_iter, max_train_iters)

    def _sample_student_sigma(self, batch_size, device, dtype):
        step_idx = self._sample_synced_int(0, len(self.denoising_sigmas))
        sigma = self.denoising_sigmas[step_idx].to(device=device, dtype=dtype)
        return sigma.expand(int(batch_size)), step_idx

    def _predict_x0(self, model, latents, sigma, condition):
        velocity = self._predict_velocity(model, latents, sigma, condition)
        sigma = self.scheduler._expand_to_ndim(sigma, latents.ndim)
        return latents - sigma * velocity

    def _predict_teacher_x0(self, latents, sigma, condition, negative_condition):
        velocity_teacher_cond = self._predict_velocity(self.teacher_model, latents, sigma, condition)
        if self.guidance_scale <= 1 or negative_condition is None:
            velocity_teacher = velocity_teacher_cond
        else:
            velocity_teacher_uncond = self._predict_velocity(self.teacher_model, latents, sigma, negative_condition)
            velocity_teacher = self._do_cfg(velocity_teacher_cond, velocity_teacher_uncond, self.guidance_scale, self.cfg_norm)
        sigma = self.scheduler._expand_to_ndim(sigma, latents.ndim)
        return latents - sigma * velocity_teacher

    def _sample_real_noised_student_input(self, real_latent):
        batch_size = real_latent.shape[0]
        student_noise = torch.randn_like(real_latent, dtype=torch.float32)
        student_sigma, _ = self._sample_student_sigma(batch_size, device=real_latent.device, dtype=self.running_dtype)
        input_student = self.scheduler.add_noise(real_latent, student_noise, student_sigma)
        return input_student, student_sigma

    def _student_generated_x0(self, real_latent, condition, grad_enabled):
        input_student, student_sigma = self._sample_real_noised_student_input(real_latent)
        self.model.transformer.train()
        context = torch.enable_grad if grad_enabled else torch.no_grad
        with context():
            generated = self._predict_x0(self.model, input_student, student_sigma, condition)
        return generated.to(dtype=self.running_dtype)

    def _teacher_score_x0(self, generated, real_latent):
        if self.dfd_real_replace_prob <= 0 or not self._sample_synced_bool(self.dfd_real_replace_prob):
            return generated

        if self.dfd_post_train:
            real_sample = real_latent.to(device=generated.device, dtype=generated.dtype)
            return generated + (real_sample - generated).detach()

        gen_idx = self._sample_synced_int(0, generated.shape[0])
        real_idx = self._sample_synced_int(0, real_latent.shape[0])
        teacher_score = generated.clone()
        real_sample = real_latent[real_idx].to(device=generated.device, dtype=generated.dtype)
        teacher_score[gen_idx] = teacher_score[gen_idx] + (real_sample - generated[gen_idx]).detach()
        return teacher_score

    def forward_loss(self, latent_shape, conditions, stage, sample=None):
        del latent_shape
        condition, negative_condition = conditions
        real_latent = self._encode_dfd_real_latent(sample, self._latent_shape(sample))

        generated = self._student_generated_x0(real_latent, condition, grad_enabled=(stage != "fake"))
        score_sigma = self._sample_score_sigma(
            real_latent.shape[0],
            denoised_timestep_from=None,
            denoised_timestep_to=None,
            device=self.model.device,
            dtype=self.running_dtype,
        )
        score_noise = torch.randn_like(real_latent, dtype=torch.float32)

        with torch.no_grad():
            perturbed_generated = self.scheduler.add_noise(generated, score_noise, score_sigma)

        if stage == "fake":
            self.fake_model.transformer.train()
            velocity_fake = self._predict_velocity(self.fake_model, perturbed_generated, score_sigma, condition)
            velocity_gt = self.scheduler.build_train_gt(generated.float(), score_noise)
            return F.mse_loss(velocity_fake.float(), velocity_gt.float(), reduction="mean")

        with torch.no_grad():
            self.fake_model.transformer.eval()
            self.teacher_model.transformer.eval()

            teacher_score_x0 = self._teacher_score_x0(generated, real_latent)
            perturbed_teacher_score = self.scheduler.add_noise(teacher_score_x0, score_noise, score_sigma)

            if self.dfd_use_teacher_as_fake_score:
                x_pred_fake = self._predict_teacher_x0(perturbed_generated, score_sigma, condition, None)
            else:
                fake_input = perturbed_teacher_score if self.dfd_forward_kl else perturbed_generated
                x_pred_fake = self._predict_x0(self.fake_model, fake_input, score_sigma, condition)

            x_pred_teacher = self._predict_teacher_x0(perturbed_teacher_score, score_sigma, condition, negative_condition)

        return self._dmd_loss(generated, x_pred_fake, x_pred_teacher)
