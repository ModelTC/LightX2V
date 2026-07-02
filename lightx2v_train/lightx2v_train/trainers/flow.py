import os

import torch
from loguru import logger

from lightx2v_train.runtime.distributed import barrier, get_world_size, is_main_process, reduce_mean
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value, sync_sequence_parallel_gradients
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .base import BaseTrainer


@TRAINER_REGISTER("flow")
class FlowMatchingTrainer(BaseTrainer):
    trainer_name = "flow"

    def compute_loss_on_sample(self, sample):
        with torch.no_grad():
            latent = self.model.encode_to_latent(sample)
            latent = broadcast_sequence_parallel_value(latent)
            n = latent.shape[0]
            noise = torch.randn_like(latent, dtype=self.running_dtype)
            noise = broadcast_sequence_parallel_value(noise)
            latent_hw = (latent.shape[-2], latent.shape[-1])
            timestep_or_sigma = self.noise_scheduler.sample_timestep_or_sigma(n, latent_hw=latent_hw)
            timestep_or_sigma = broadcast_sequence_parallel_value(timestep_or_sigma)
            noisy_latent = self.noise_scheduler.add_noise(latent, noise, timestep_or_sigma)
            condition = self.model.encode_condition(sample)
            condition = broadcast_sequence_parallel_value(condition)

        denoiser_input = self.model.prepare_denoiser_input(noisy_latent)
        prediction = self.model.denoise(denoiser_input, timestep_or_sigma, condition)
        prediction = self.model.postprocess_denoiser_output(prediction, denoiser_input)

        target = self.noise_scheduler.build_train_gt(latent, noise)
        loss = torch.mean(((prediction.float() - target.float()) ** 2).reshape(target.shape[0], -1), dim=1)
        return loss.mean()

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        max_train_iters = self.max_train_iters
        grad_accum_iters = self.gradient_accumulation_iters
        max_grad_norm = self.max_grad_norm
        save_every_iters = self.save_every_iters
        save_total_limit = self.save_total_limit
        grad_accum_counter = 0
        running_loss = 0.0

        logger.info(
            "[train] start method={} train_type={} iter={}/{} world_size={} grad_accum={} train_log_every_iters={}",
            self.trainer_name,
            self.train_type,
            current_iter,
            max_train_iters,
            get_world_size(),
            grad_accum_iters,
            self.train_log_every_iters,
        )
        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_eval)
            if current_iter == 0:
                self.run_inference(current_iter)

        epoch = 0
        while current_iter < max_train_iters:
            sampler = getattr(self.dataloader_train, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for sample in self.dataloader_train:
                sync_grad = (grad_accum_counter + 1) % grad_accum_iters == 0
                self._set_gradient_sync(sync_grad)

                loss = self.compute_loss_on_sample(sample)
                (loss / grad_accum_iters).backward()
                running_loss += loss.item() / grad_accum_iters

                grad_accum_counter += 1
                if grad_accum_counter % grad_accum_iters != 0:
                    continue

                self._after_backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_params, max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                current_iter += 1
                display_loss = reduce_mean(running_loss)
                current_lr = self.lr_scheduler.get_last_lr()[0]
                if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= max_train_iters:
                    logger.info("[train] iter={}/{} loss={:.6f} lr={:.8f}", current_iter, max_train_iters, display_loss, current_lr)
                running_loss = 0.0

                if save_every_iters and current_iter % save_every_iters == 0:
                    self.save_checkpoint(current_iter, save_total_limit)

                if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                    self.run_inference(current_iter)

                if current_iter >= max_train_iters:
                    break

            epoch += 1

        logger.info("[train] finished iter={}/{}", current_iter, max_train_iters)

    def _after_backward(self):
        sync_sequence_parallel_gradients(self.trainable_params)
