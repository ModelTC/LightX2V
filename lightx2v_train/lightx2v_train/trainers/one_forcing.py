import os

import torch
import torch.nn.functional as F
from loguru import logger

from lightx2v_train.runtime.distributed import barrier, get_world_size, is_main_process, reduce_mean
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .dmd import VideoArDmdTrainer


@TRAINER_REGISTER("one_forcing")
class OneForcingTrainer(VideoArDmdTrainer):
    """One-Forcing: causal self-rollout DMD with a shared fake-score GAN critic."""

    trainer_name = "one_forcing"

    def __init__(self, config):
        super().__init__(config)
        if self.fake_train_type != "full":
            raise ValueError("one_forcing currently requires training.fake.train_type=full so the GAN head is trainable.")

        self.one_forcing_config = self.training_config.get("one_forcing", {})
        self.gan_g_weight = float(self.one_forcing_config.get("gan_g_weight", 0.03))
        self.gan_d_weight = float(self.one_forcing_config.get("gan_d_weight", 0.03))
        self.gan_feature_layers = tuple(self.one_forcing_config.get("gan_feature_layers", (21, 29)))
        self.gan_hidden_dim = self.one_forcing_config.get("gan_hidden_dim")
        self.gan_ffn_dim = self.one_forcing_config.get("gan_ffn_dim")
        self.gan_num_heads = self.one_forcing_config.get("gan_num_heads")
        self.concat_time_embedding = bool(self.one_forcing_config.get("concat_time_embedding", False))
        self.real_latent_key = self.one_forcing_config.get("real_latent_key", "latent")

        if self.gan_g_weight < 0 or self.gan_d_weight < 0:
            raise ValueError("One-Forcing GAN weights must be non-negative.")
        if self.gan_g_weight == 0 and self.gan_d_weight == 0:
            logger.warning("Both One-Forcing GAN weights are zero; training reduces to video_ar_dmd.")

    def _setup_trainable_model(self, model, role="student"):
        if role == "fake":
            model.add_gan_classifier(
                feature_layers=self.gan_feature_layers,
                hidden_dim=self.gan_hidden_dim,
                ffn_dim=self.gan_ffn_dim,
                num_heads=self.gan_num_heads,
                concat_time_embedding=self.concat_time_embedding,
            )
        super()._setup_trainable_model(model, role=role)

    def _real_latent(self, sample):
        if self.real_latent_key not in sample:
            raise KeyError(
                f"One-Forcing requires real video latents in sample[{self.real_latent_key!r}]. "
                "Use data.train.name=causal_forcing_lmdb_dataset or another dataset with this key."
            )
        latent = sample[self.real_latent_key]
        if not torch.is_tensor(latent) or latent.ndim != 5:
            shape = None if not torch.is_tensor(latent) else tuple(latent.shape)
            raise ValueError(f"Real latent must be a [B,C,F,H,W] tensor, got {shape}.")
        return latent.to(device=self.model.device, dtype=self.running_dtype)

    @staticmethod
    def _match_real_latent(real_latent, generated):
        if real_latent.shape[2] > generated.shape[2]:
            real_latent = real_latent[:, :, -generated.shape[2] :]
        if real_latent.shape != generated.shape:
            raise ValueError(
                f"Real latent shape {tuple(real_latent.shape)} does not match generated shape "
                f"{tuple(generated.shape)}. Configure matching frame count and resolution."
            )
        return real_latent.detach()

    def _sample_gan_sigma(self, generated, denoised_timestep_from, denoised_timestep_to):
        return self._sample_score_sigma(
            generated.shape[0],
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            device=generated.device,
            dtype=self.running_dtype,
        )

    def _gan_generator_loss(self, generated, condition, denoised_timestep_from, denoised_timestep_to):
        if self.gan_g_weight == 0:
            zero = generated.new_zeros(())
            return zero, zero

        sigma = self._sample_gan_sigma(generated, denoised_timestep_from, denoised_timestep_to)
        noisy_fake = self.scheduler.add_noise(generated, torch.randn_like(generated), sigma)
        fake_logits = self.fake_model.classify_latents(noisy_fake, sigma, condition)
        raw_loss = F.softplus(-fake_logits.float()).mean() * self.gan_g_weight

        # Differentiate only to the generated latent. The shared fake-score parameters
        # are updated exclusively in the critic stage, as in the original algorithm.
        generated_grad = torch.autograd.grad(raw_loss, generated, retain_graph=False)[0]
        loss_proxy = (generated.float() * generated_grad.detach().float()).sum()
        return loss_proxy, raw_loss.detach()

    def _gan_discriminator_loss(self, generated, real_latent, condition, denoised_timestep_from, denoised_timestep_to):
        if self.gan_d_weight == 0:
            return generated.new_zeros(())

        sigma = self._sample_gan_sigma(generated, denoised_timestep_from, denoised_timestep_to)
        shared_noise = torch.randn_like(generated)
        noisy_fake = self.scheduler.add_noise(generated.detach(), shared_noise, sigma)
        noisy_real = self.scheduler.add_noise(real_latent, shared_noise, sigma)
        combined = torch.cat((noisy_fake, noisy_real), dim=0)
        combined_sigma = torch.cat((sigma, sigma), dim=0)
        combined_condition = {
            key: torch.cat((value, value), dim=0) if torch.is_tensor(value) else value
            for key, value in condition.items()
        }
        logits = self.fake_model.classify_latents(combined, combined_sigma, combined_condition)
        fake_logits, real_logits = logits.chunk(2, dim=0)
        return self.gan_d_weight * (
            F.softplus(fake_logits.float()).mean() + F.softplus(-real_logits.float()).mean()
        )

    def forward_loss(self, latent_shape, conditions, real_latent, stage):
        condition, negative_condition = conditions
        generated, denoised_timestep_from, denoised_timestep_to = self.run_back_simulation(
            condition,
            latent_shape,
            grad_enabled=(stage != "fake"),
        )
        real_latent = self._match_real_latent(real_latent, generated)

        sigma = self._sample_score_sigma(
            latent_shape[0],
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            device=self.model.device,
            dtype=self.running_dtype,
        )
        noise = torch.randn(latent_shape, device=self.model.device, dtype=torch.float32)
        with torch.no_grad():
            renoised_xt = self.scheduler.add_noise(generated, noise, sigma)

        if stage == "fake":
            self.fake_model.transformer.train()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition)
            velocity_gt = self.scheduler.build_train_gt(generated.float(), noise)
            loss_fake = F.mse_loss(velocity_fake.float(), velocity_gt.float(), reduction="mean")
            loss_gan_d = self._gan_discriminator_loss(
                generated,
                real_latent,
                condition,
                denoised_timestep_from,
                denoised_timestep_to,
            )
            return {"loss": loss_fake + loss_gan_d, "fake": loss_fake.detach(), "gan_d": loss_gan_d.detach()}

        with torch.no_grad():
            self.fake_model.transformer.eval()
            self.teacher_model.transformer.eval()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition)
            velocity_teacher_cond = self._predict_velocity(self.teacher_model, renoised_xt, sigma, condition)
            if negative_condition is None:
                velocity_teacher = velocity_teacher_cond
            else:
                velocity_teacher_uncond = self._predict_velocity(self.teacher_model, renoised_xt, sigma, negative_condition)
                velocity_teacher = self._do_cfg(
                    velocity_teacher_cond,
                    velocity_teacher_uncond,
                    self.guidance_scale,
                    self.cfg_norm,
                )

            expanded_sigma = self.scheduler._expand_to_ndim(sigma, renoised_xt.ndim)
            x_pred_fake = renoised_xt - expanded_sigma * velocity_fake
            x_pred_teacher = renoised_xt - expanded_sigma * velocity_teacher

        loss_dmd = self._dmd_loss(generated, x_pred_fake, x_pred_teacher)
        loss_gan_proxy, loss_gan_g = self._gan_generator_loss(
            generated,
            condition,
            denoised_timestep_from,
            denoised_timestep_to,
        )
        return {"loss": loss_dmd + loss_gan_proxy, "dmd": loss_dmd.detach(), "gan_g": loss_gan_g}

    def _train_one_stage(self, samples, stage, grad_accum_iters):
        if stage == "student":
            optimizer, scheduler, params = self.optimizer, self.lr_scheduler, self.trainable_params
            set_sync = self._set_student_gradient_sync
        elif stage == "fake":
            optimizer, scheduler, params = self.fake_optimizer, self.fake_lr_scheduler, self.fake_trainable_params
            set_sync = self._set_fake_gradient_sync
        else:
            raise ValueError(f"Unsupported One-Forcing training stage: {stage}")

        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        for micro_idx in range(grad_accum_iters):
            sample = next(samples)
            conditions = self._encode_conditions(sample)
            real_latent = self._real_latent(sample)
            latent_shape = tuple(real_latent.shape)
            set_sync(micro_idx == grad_accum_iters - 1)
            result = self.forward_loss(latent_shape, conditions, real_latent, stage=stage)
            (result["loss"] / grad_accum_iters).backward()
            for key, value in result.items():
                if key != "loss":
                    metrics[key] = metrics.get(key, 0.0) + value.item() / grad_accum_iters

        self._sync_sequence_parallel_grads(params)
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return metrics

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        logger.info(
            "[train] start method=one_forcing iter={}/{} world_size={} grad_accum={} fake_update_ratio={} gan_g_weight={} gan_d_weight={}",
            current_iter,
            self.max_train_iters,
            get_world_size(),
            self.gradient_accumulation_iters,
            self.fake_update_ratio,
            self.gan_g_weight,
            self.gan_d_weight,
        )
        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_eval)
            if current_iter == 0:
                self.run_inference(current_iter)

        samples = self._iter_train_samples()
        last_student_metrics = {}
        while current_iter < self.max_train_iters:
            if current_iter % self.fake_update_ratio == 0:
                last_student_metrics = self._train_one_stage(
                    samples,
                    stage="student",
                    grad_accum_iters=max(1, int(self.gradient_accumulation_iters)),
                )
            fake_metrics = self._train_one_stage(
                samples,
                stage="fake",
                grad_accum_iters=max(1, int(self.gradient_accumulation_iters)),
            )
            current_iter += 1

            if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= self.max_train_iters:
                logger.info(
                    "[train] iter={}/{} dmd={:.6f} gan_g={:.6f} fake={:.6f} gan_d={:.6f} student_lr={:.8f} fake_lr={:.8f}",
                    current_iter,
                    self.max_train_iters,
                    reduce_mean(last_student_metrics.get("dmd", 0.0)),
                    reduce_mean(last_student_metrics.get("gan_g", 0.0)),
                    reduce_mean(fake_metrics.get("fake", 0.0)),
                    reduce_mean(fake_metrics.get("gan_d", 0.0)),
                    self.lr_scheduler.get_last_lr()[0],
                    self.fake_lr_scheduler.get_last_lr()[0],
                )

            if self.save_every_iters and current_iter % self.save_every_iters == 0:
                self.save_checkpoint(current_iter, self.save_total_limit)
            if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                self.run_inference(current_iter)

        logger.info("[train] finished iter={}/{}", current_iter, self.max_train_iters)
