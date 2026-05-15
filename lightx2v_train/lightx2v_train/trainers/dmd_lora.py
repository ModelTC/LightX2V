import os

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from lightx2v_train.model_zoo import build_model
from lightx2v_train.schedulers import DMDFlowMatchingScheduler
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .lora import LoraTrainer


@TRAINER_REGISTER("dmd_lora")
class DmdLoraTrainer(LoraTrainer):
    def __init__(self, config):
        super().__init__(config)
        fake_config = self.training_config.get("fake", {})
        fake_optimizer_config = fake_config.get("optimizer", {})
        self.fake_optimizer_learning_rate = fake_optimizer_config.get("learning_rate", self.optimizer_learning_rate)
        self.fake_optimizer_adam_beta1 = fake_optimizer_config.get("adam_beta1", self.optimizer_adam_beta1)
        self.fake_optimizer_adam_beta2 = fake_optimizer_config.get("adam_beta2", self.optimizer_adam_beta2)
        self.fake_optimizer_weight_decay = fake_optimizer_config.get("weight_decay", self.optimizer_weight_decay)
        self.fake_optimizer_adam_epsilon = fake_optimizer_config.get("adam_epsilon", self.optimizer_adam_epsilon)

        self.dmd_config = self.training_config.get("dmd", {})
        self.num_inference_steps = int(self.dmd_config.get("num_inference_steps", 4))
        self.fake_update_ratio = int(self.dmd_config.get("fake_update_ratio", 1))
        self.guidance_scale = float(self.dmd_config.get("guidance_scale", 3.0))
        self.negative_prompt = self.dmd_config.get("negative_prompt", " ")
        self.cfg_norm = self.dmd_config.get("cfg_norm", "layer_norm")

    def setup(self):
        super().setup()
        self.fake_model = build_model(self.config)
        self.fake_model.load_components(transformer_only=True, reference_model=self.model)
        self.fake_model.add_lora(self.lora_rank, self.lora_alpha, self.lora_target_modules)
        self.fake_model.set_lora_trainable()
        if self.gradient_checkpointing:
            self.fake_model.enable_gradient_checkpointing()

        self.teacher_model = build_model(self.config)
        self.teacher_model.load_components(transformer_only=True, reference_model=self.model)
        self.teacher_model.transformer.requires_grad_(False)
        self.teacher_model.transformer.eval()

        self.fake_optimizer = torch.optim.AdamW(
            self.fake_model.trainable_parameters(),
            lr=self.fake_optimizer_learning_rate,
            betas=(self.fake_optimizer_adam_beta1, self.fake_optimizer_adam_beta2),
            weight_decay=self.fake_optimizer_weight_decay,
            eps=self.fake_optimizer_adam_epsilon,
        )
        self.fake_lr_scheduler = get_scheduler(
            self.lr_scheduler_name,
            optimizer=self.fake_optimizer,
            num_warmup_steps=0,
            num_training_steps=max(1, self.max_train_iters * self.fake_update_ratio),
        )

        self.scheduler = DMDFlowMatchingScheduler(self.config, self.dmd_config)

        print(f"[dmd_lora] student trainable params={self._count_trainable(self.model.transformer)}")
        print(f"[dmd_lora] fake trainable params={self._count_trainable(self.fake_model.transformer)}")

    @staticmethod
    def _count_trainable(module):
        return sum(1 for param in module.parameters() if param.requires_grad)

    @staticmethod
    def _do_cfg(cond_pred, uncond_pred, cfg_scale, cfg_norm):
        pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        if cfg_norm in (None, "none"):
            return pred
        if cfg_norm == "layer_norm":
            cond_norm = torch.norm(cond_pred, dim=-1, keepdim=True)
            pred_norm = torch.norm(pred, dim=-1, keepdim=True)
            return pred * (cond_norm / torch.clamp(pred_norm, min=1e-12))
        if cfg_norm == "scalar":
            cond_norm = torch.norm(cond_pred)
            pred_norm = torch.norm(pred)
            return pred * min(1.0, (cond_norm / torch.clamp(pred_norm, min=1e-12)).item())
        raise ValueError(f"Unsupported cfg_norm: {cfg_norm}")

    @staticmethod
    def _dmd_loss(latents, x_pred_fake_flow, x_pred_teacher):
        with torch.no_grad():
            grad = x_pred_fake_flow - x_pred_teacher
            dims = tuple(range(1, latents.ndim))
            normalizer = torch.abs(latents - x_pred_teacher).mean(dim=dims, keepdim=True)
            grad = torch.nan_to_num(grad / normalizer)
        return 0.5 * F.mse_loss(latents.float(), (latents.float() - grad.float()).detach(), reduction="mean")

    def _latent_shape(self, sample):
        image = sample["target_image"]
        batch_size = image.shape[0]
        latent_channels = getattr(self.model.vae.config, "z_dim", None)
        if latent_channels is None:
            latent_channels = self.model.transformer.config.in_channels // 4
        return (
            batch_size,
            int(latent_channels),
            1,
            image.shape[-2] // self.model.vae_scale_factor,
            image.shape[-1] // self.model.vae_scale_factor,
        )

    def _encode_conditions(self, sample):
        prompt = sample["prompt"]
        if isinstance(prompt, str):
            negative_prompt = self.negative_prompt
        else:
            negative_prompt = [self.negative_prompt] * len(prompt)
        with torch.no_grad():
            condition = self.model.encode_prompt_condition(prompt)
            negative_condition = self.model.encode_prompt_condition(negative_prompt)
        return condition, negative_condition

    def _predict_velocity(self, model, latents, sigma, condition):
        denoiser_input = model.prepare_denoiser_input(latents)
        prediction = model.denoise(denoiser_input, sigma, condition)
        prediction = model.postprocess_denoiser_output(prediction, denoiser_input)
        return prediction

    def sample_initial_latents(self, latent_shape):
        return torch.randn(latent_shape, device=self.model.device, dtype=self.running_dtype)

    def sample_end_step(self):
        return int(torch.randint(0, self.num_inference_steps, (1,), device=self.model.device).item())

    def run_back_simulation(self, condition, latent_shape, end_step_idx, grad_enabled, xt=None):
        self.scheduler.set_timesteps(self.num_inference_steps)
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        x0 = None
        self.model.transformer.train()
        for idx in range(end_step_idx + 1):
            sigma = self.scheduler.sigma_at(idx, latent_shape[0], device=self.model.device, dtype=self.running_dtype)
            context = torch.enable_grad if (grad_enabled and idx == end_step_idx) else torch.no_grad
            with context():
                velocity = self._predict_velocity(self.model, xt, sigma, condition)
            xt, x0 = self.scheduler.step_by_index(velocity, idx, xt)
        return x0

    def forward_loss(self, sample, stage):
        condition, negative_condition = self._encode_conditions(sample)
        latent_shape = self._latent_shape(sample)
        end_step_idx = self.sample_end_step()
        xt_start = self.sample_initial_latents(latent_shape)
        x0_ref = self.run_back_simulation(condition, latent_shape, end_step_idx, grad_enabled=False, xt=xt_start)

        sigma = self.scheduler.sample_renoise_sigma(latent_shape[0], device=self.model.device, dtype=self.running_dtype)
        noise = torch.randn(latent_shape, device=self.model.device, dtype=torch.float32)
        renoised_xt = self.scheduler.add_noise(x0_ref, noise, sigma)
        velocity_gt = self.scheduler.build_train_gt(x0_ref.float(), noise)

        if stage == "fake":
            self.fake_model.transformer.train()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition)
            return F.mse_loss(velocity_fake.float(), velocity_gt.float(), reduction="mean")

        with torch.no_grad():
            self.fake_model.transformer.eval()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition)
            velocity_teacher_cond = self._predict_velocity(self.teacher_model, renoised_xt, sigma, condition)
            velocity_teacher_uncond = self._predict_velocity(self.teacher_model, renoised_xt, sigma, negative_condition)
            velocity_teacher = self._do_cfg(velocity_teacher_cond, velocity_teacher_uncond, self.guidance_scale, self.cfg_norm)

        zeros = torch.zeros_like(sigma)
        x_pred_fake = self.scheduler.euler_step(renoised_xt, velocity_fake, sigma, zeros)
        x_pred_teacher = self.scheduler.euler_step(renoised_xt, velocity_teacher, sigma, zeros)
        x0 = self.run_back_simulation(condition, latent_shape, end_step_idx, grad_enabled=True, xt=xt_start)
        return self._dmd_loss(x0, x_pred_fake, x_pred_teacher)

    def train(self):
        self.setup()
        os.makedirs(self.output_train_dir, exist_ok=True)

        max_train_iters = self.max_train_iters
        fake_update_ratio = self.fake_update_ratio
        max_grad_norm = self.max_grad_norm
        save_every_iters = self.save_every_iters
        save_total_limit = self.save_total_limit
        current_iter = 0
        running_dmd = 0.0
        running_fake = 0.0

        progress = tqdm(total=max_train_iters, desc="DMD-LoRA iterations")

        while current_iter < max_train_iters:
            for sample in self.dataloader_train:
                loss_dmd = self.forward_loss(sample, stage="generator")
                loss_dmd.backward()
                torch.nn.utils.clip_grad_norm_(self.model.transformer.parameters(), max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                running_dmd += loss_dmd.item()

                fake_loss = 0.0
                for _ in range(fake_update_ratio):
                    loss_fake = self.forward_loss(sample, stage="fake")
                    loss_fake.backward()
                    torch.nn.utils.clip_grad_norm_(self.fake_model.transformer.parameters(), max_grad_norm)
                    self.fake_optimizer.step()
                    self.fake_lr_scheduler.step()
                    self.fake_optimizer.zero_grad(set_to_none=True)
                    fake_loss += loss_fake.item()
                running_fake += fake_loss / fake_update_ratio

                current_iter += 1
                progress.update(1)
                progress.set_postfix(
                    dmd=running_dmd,
                    fake=running_fake,
                    lr=self.lr_scheduler.get_last_lr()[0],
                )
                running_dmd = 0.0
                running_fake = 0.0

                if save_every_iters and current_iter % save_every_iters == 0:
                    self.save_checkpoint(current_iter, save_total_limit)

                if current_iter >= max_train_iters:
                    break

        progress.close()
