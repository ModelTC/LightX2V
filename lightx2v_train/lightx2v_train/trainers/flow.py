import os

import torch
from loguru import logger
from torch import Tensor

from lightx2v_train.runtime.distributed import barrier, get_world_size, is_main_process, reduce_mean
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value, sync_sequence_parallel_gradients
from lightx2v_train.utils.ltx2_native import (
    AudioLatentShape,
    AudioPatchifier,
    Modality,
    SpatioTemporalScaleFactors,
    VideoLatentPatchifier,
    VideoLatentShape,
    get_pixel_coords,
)
from lightx2v_train.utils.registry import TRAINER_REGISTER
from lightx2v_train.utils.sample import first_scalar, nested_to_device, sample_condition, sample_input

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

        denoiser_input = self.model.prepare_denoiser_input(noisy_latent, condition=condition)
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


@TRAINER_REGISTER("ltx_t2av_flow")
class LTX2T2AVFlowTrainer(FlowMatchingTrainer):
    trainer_name = "ltx_t2av_flow"

    def __init__(self, config):
        super().__init__(config)
        ltx2_config = self.training_config.get("ltx2", {})
        self.video_loss_weight = float(ltx2_config.get("video_loss_weight", 1.0))
        self.audio_loss_weight = float(ltx2_config.get("audio_loss_weight", 1.0))
        self.default_fps = float(ltx2_config.get("default_fps", 24.0))
        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)

    def compute_loss_on_sample(self, sample):
        video_data = nested_to_device(sample_input(sample, "video_latents"), self.model.device)
        audio_data = nested_to_device(sample_input(sample, "audio_latents"), self.model.device)
        conditions = sample_condition(sample, "positive")
        if video_data is None or audio_data is None or conditions is None:
            raise KeyError("ltx_t2av_flow expects inputs.video_latents, inputs.audio_latents and conditioning.positive.")

        video_latents = self._video_latents(video_data)
        audio_latents = self._audio_latents(audio_data)
        video_tokens = self.video_patchifier.patchify(video_latents)
        audio_tokens = self.audio_patchifier.patchify(audio_latents)

        video_context, audio_context, context_mask = self.model.prepare_text_condition(conditions)
        video_input, video_target, video_timesteps, video_loss_mask, video_sigmas = self._initialize_noisy_target(video_tokens)
        audio_input, audio_target, audio_timesteps, audio_loss_mask, audio_sigmas = self._initialize_noisy_target(audio_tokens, sigmas=video_sigmas)

        video_positions = self._get_video_positions(
            num_frames=int(first_scalar(video_data.get("num_frames"), video_latents.shape[2])),
            height=int(first_scalar(video_data.get("height"), video_latents.shape[3])),
            width=int(first_scalar(video_data.get("width"), video_latents.shape[4])),
            batch_size=video_tokens.shape[0],
            fps=float(first_scalar(video_data.get("fps"), self.default_fps)),
            device=video_tokens.device,
        )
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_tokens.shape[1],
            batch_size=audio_tokens.shape[0],
            device=audio_tokens.device,
        )

        video_modality = Modality(
            enabled=True,
            latent=video_input,
            sigma=video_sigmas,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_context,
            context_mask=context_mask,
        )
        audio_modality = Modality(
            enabled=True,
            latent=audio_input,
            sigma=audio_sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_context,
            context_mask=context_mask,
        )

        with self.model.transformer_forward_context():
            video_pred, audio_pred = self.model.denoiser_module()(
                video=video_modality,
                audio=audio_modality,
                perturbations=None,
            )

        video_loss = self._masked_mse(video_pred, video_target, video_loss_mask)
        audio_loss = self._masked_mse(audio_pred, audio_target, audio_loss_mask)
        loss = self.video_loss_weight * video_loss + self.audio_loss_weight * audio_loss
        return loss.mean()

    def _video_latents(self, video_data):
        latents = video_data["latents"].to(device=self.model.device, dtype=self.running_dtype)
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        if latents.ndim != 5:
            raise ValueError(f"LTX2 video latents must be [B,C,F,H,W] or [C,F,H,W], got shape={tuple(latents.shape)}")
        return latents.contiguous()

    def _audio_latents(self, audio_data):
        latents = audio_data["latents"].to(device=self.model.device, dtype=self.running_dtype)
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)
        if latents.ndim != 4:
            raise ValueError(f"LTX2 audio latents must be [B,C,T,F] or [C,T,F], got shape={tuple(latents.shape)}")
        return latents.contiguous()

    def _initialize_noisy_target(self, latents: Tensor, sigmas=None):
        batch_size, seq_len, _ = latents.shape
        if sigmas is None:
            sigmas = self.noise_scheduler.sample_timestep_or_sigma(batch_size, seq_len=seq_len)
        sigmas = sigmas.to(device=latents.device, dtype=latents.dtype)
        noise = torch.randn_like(latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_latents = (1 - sigmas_expanded) * latents + sigmas_expanded * noise
        targets = noise - latents
        timesteps = sigmas.view(-1, 1).expand(batch_size, seq_len).clone()
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=latents.device)
        return noisy_latents, targets, timesteps, loss_mask, sigmas

    def _get_video_positions(self, num_frames, height, width, batch_size, fps, device):
        latent_coords = self.video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                frames=num_frames,
                height=height,
                width=width,
                batch=batch_size,
                channels=128,
            ),
            device=device,
        )
        pixel_coords = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
        ).float()
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps
        return pixel_coords

    def _get_audio_positions(self, num_time_steps, batch_size, device):
        return self.audio_patchifier.get_patch_grid_bounds(
            output_shape=AudioLatentShape(
                frames=num_time_steps,
                mel_bins=16,
                batch=batch_size,
                channels=8,
            ),
            device=device,
        )

    @staticmethod
    def _masked_mse(pred, targets, loss_mask):
        target_len = targets.shape[1]
        pred = pred[:, -target_len:, :]
        mask = loss_mask[:, -target_len:].unsqueeze(-1).float()
        loss = (pred.float() - targets.float()).pow(2) * mask
        return loss.mean(dim=(-2, -1)) / mask.mean(dim=(-2, -1)).clamp(min=1e-8)
