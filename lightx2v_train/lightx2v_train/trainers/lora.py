import os

import torch
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from lightx2v_train.runtime import prune_checkpoints
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .base import BaseTrainer


@TRAINER_REGISTER("lora")
class LoraTrainer(BaseTrainer):
    def get_configs(self):
        training_config = self.config["training"]

        lora_config = training_config.get("lora", {})
        self.lora_rank = lora_config.get("rank", 16)
        self.lora_alpha = lora_config.get("alpha", self.lora_rank)
        self.lora_target_modules = lora_config.get("target_modules")

        self.gradient_checkpointing = training_config.get("gradient_checkpointing", True)

        optimizer_config = training_config.get("optimizer", {})
        self.optimizer_learning_rate = optimizer_config.get("learning_rate", 1e-4)
        self.optimizer_adam_beta1 = optimizer_config.get("adam_beta1", 0.9)
        self.optimizer_adam_beta2 = optimizer_config.get("adam_beta2", 0.999)
        self.optimizer_weight_decay = optimizer_config.get("weight_decay", 0.01)
        self.optimizer_adam_epsilon = optimizer_config.get("adam_epsilon", 1e-8)

        self.lr_scheduler_name = training_config.get("lr_scheduler", "constant")
        self.lr_warmup_iters = training_config["lr_warmup_iters"]
        self.max_train_iters = training_config["max_train_iters"]

        self.output_dir = training_config["output_dir"]
        self.gradient_accumulation_iters = training_config["gradient_accumulation_iters"]
        self.max_grad_norm = training_config.get("max_grad_norm", 1.0)
        self.save_every_iters = training_config["save_every_iters"]
        self.save_total_limit = training_config["save_total_limit"]

    def setup(self):
        self.get_configs()
        self.model.add_lora(self.lora_rank, self.lora_alpha, self.lora_target_modules)
        self.model.set_lora_trainable()
        if self.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        self.optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=self.optimizer_learning_rate,
            betas=(self.optimizer_adam_beta1, self.optimizer_adam_beta2),
            weight_decay=self.optimizer_weight_decay,
            eps=self.optimizer_adam_epsilon,
        )
        self.lr_scheduler = get_scheduler(
            self.lr_scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=self.lr_warmup_iters,
            num_training_steps=self.max_train_iters,
        )

    def compute_loss_on_sample(self, sample):
        with torch.no_grad():
            latent = self.model.encode_to_latent(sample)
            n = latent.shape[0]
            noise = torch.randn_like(latent, dtype=self.model.dtype)
            timesteps = self.noise_scheduler.sample_timesteps(n, latent.device)
            sigmas = self.noise_scheduler.get_sigmas(timesteps, n_dim=latent.ndim, dtype=latent.dtype)
            noisy_latent = self.noise_scheduler.add_noise(latent, noise, sigmas)
            condition = self.model.encode_condition(sample)

        denoiser_input = self.model.prepare_denoiser_input(noisy_latent, sample, condition)
        prediction = self.model.denoise(denoiser_input, timesteps, condition)
        prediction = self.model.postprocess_denoiser_output(prediction, denoiser_input)

        weighting = self.noise_scheduler.loss_weighting(sigmas)
        target = self.model.prepare_flow_matching_target(self.noise_scheduler.build_train_gt(latent, noise))
        loss = torch.mean(
            (weighting.float() * (prediction.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            dim=1,
        )
        return loss.mean()

    def train(self):
        self.setup()
        os.makedirs(self.output_dir, exist_ok=True)

        max_train_iters = self.max_train_iters
        grad_accum_iters = self.gradient_accumulation_iters
        max_grad_norm = self.max_grad_norm
        save_every_iters = self.save_every_iters
        save_total_limit = self.save_total_limit
        current_iter = 0
        grad_accum_counter = 0
        running_loss = 0.0

        progress = tqdm(total=max_train_iters, desc="Training iterations")
        while current_iter < max_train_iters:
            for sample in self.dataloader:
                loss = self.compute_loss_on_sample(sample)
                (loss / grad_accum_iters).backward()
                running_loss += loss.item() / grad_accum_iters

                grad_accum_counter += 1
                if grad_accum_counter % grad_accum_iters != 0:
                    continue

                torch.nn.utils.clip_grad_norm_(self.model.transformer.parameters(), max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                current_iter += 1
                progress.update(1)
                progress.set_postfix(loss=running_loss, lr=self.lr_scheduler.get_last_lr()[0])
                running_loss = 0.0

                if save_every_iters and current_iter % save_every_iters == 0:
                    self.save_checkpoint(current_iter, save_total_limit)

                if current_iter >= max_train_iters:
                    break

        progress.close()

    def save_checkpoint(self, iteration, save_total_limit):
        output_dir = self.output_dir
        prune_checkpoints(output_dir, save_total_limit)

        save_dir = os.path.join(output_dir, f"checkpoint-{iteration}")
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_lora_weights(save_dir)
