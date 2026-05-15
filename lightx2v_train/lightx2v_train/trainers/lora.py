import os

import torch
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from lightx2v_train.infer import build_inferencer
from lightx2v_train.runtime.checkpoint import find_latest_checkpoint, prune_checkpoints
from lightx2v_train.utils.registry import TRAINER_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .base import BaseTrainer


@TRAINER_REGISTER("lora")
class LoraTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.running_dtype = get_running_dtype(self.model_config["running_dtype"])

        lora_config = self.training_config.get("lora", {})
        self.lora_rank = lora_config.get("rank", 16)
        self.lora_alpha = lora_config.get("alpha", self.lora_rank)
        self.lora_target_modules = lora_config.get("target_modules")

        self.gradient_checkpointing = self.training_config.get("gradient_checkpointing", True)

        optimizer_config = self.training_config.get("optimizer", {})
        self.optimizer_learning_rate = optimizer_config.get("learning_rate", 1e-4)
        self.optimizer_adam_beta1 = optimizer_config.get("adam_beta1", 0.9)
        self.optimizer_adam_beta2 = optimizer_config.get("adam_beta2", 0.999)
        self.optimizer_weight_decay = optimizer_config.get("weight_decay", 0.01)
        self.optimizer_adam_epsilon = optimizer_config.get("adam_epsilon", 1e-8)

        self.lr_scheduler_name = self.training_config.get("lr_scheduler", "constant")
        self.lr_warmup_iters = self.training_config["lr_warmup_iters"]
        self.max_train_iters = self.training_config["max_train_iters"]

        self.output_train_dir = self.training_config["output_dir"]
        self.gradient_accumulation_iters = self.training_config["gradient_accumulation_iters"]
        self.max_grad_norm = self.training_config.get("max_grad_norm", 1.0)
        self.save_every_iters = self.training_config["save_every_iters"]
        self.save_total_limit = self.training_config["save_total_limit"]

        self.infer_every_iters = self.infer_config.get("infer_every_iters", None)

        resume_config = self.config.get("resume", {})
        self.auto_resume = resume_config.get("auto_resume", False)

    def setup(self, resume_ckpt_path=None):
        self.model.add_lora(self.lora_rank, self.lora_alpha, self.lora_target_modules)

        self.model.set_lora_trainable()
        if self.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        if self.infer_every_iters:
            self.inferencer = build_inferencer(self.config)
            self.inferencer.set_model(self.model)

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

        if resume_ckpt_path is not None:
            training_state_path = os.path.join(resume_ckpt_path, "training_state.pt")
            if os.path.exists(training_state_path):
                state = torch.load(training_state_path, map_location="cpu", weights_only=False)
                self.model.load_lora_weights_for_resume(resume_ckpt_path)
                self.optimizer.load_state_dict(state["optimizer"])
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])
                print(f"Restored training state from {training_state_path}")
            else:
                print(f"Warning: training_state.pt not found in {resume_ckpt_path}. States not restored.")

    def compute_loss_on_sample(self, sample):
        with torch.no_grad():
            latent = self.model.encode_to_latent(sample)
            n = latent.shape[0]
            noise = torch.randn_like(latent, dtype=self.running_dtype)
            timestep_or_sigma = self.noise_scheduler.sample_timestep_or_sigma(n)
            noisy_latent = self.noise_scheduler.add_noise(latent, noise, timestep_or_sigma)
            condition = self.model.encode_condition(sample)

        denoiser_input = self.model.prepare_denoiser_input(noisy_latent)
        prediction = self.model.denoise(denoiser_input, timestep_or_sigma, condition)
        prediction = self.model.postprocess_denoiser_output(prediction, denoiser_input)

        target = self.noise_scheduler.build_train_gt(latent, noise)
        loss = torch.mean(((prediction.float() - target.float()) ** 2).reshape(target.shape[0], -1), dim=1)
        return loss.mean()

    def _resolve_resume(self):
        if not self.auto_resume:
            return None, 0
        ckpt_path, current_iter = find_latest_checkpoint(self.output_train_dir)
        if ckpt_path is None:
            print(f"Auto-resume enabled but no checkpoint found in '{self.output_train_dir}'. Starting from scratch.")
        else:
            print(f"Auto-resuming from checkpoint: {ckpt_path} (iteration {current_iter})")
        return ckpt_path, current_iter

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        os.makedirs(self.output_train_dir, exist_ok=True)

        max_train_iters = self.max_train_iters
        grad_accum_iters = self.gradient_accumulation_iters
        max_grad_norm = self.max_grad_norm
        save_every_iters = self.save_every_iters
        save_total_limit = self.save_total_limit
        grad_accum_counter = 0
        running_loss = 0.0

        progress = tqdm(total=max_train_iters, desc="Training iterations", initial=current_iter)
        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_eval)
            if current_iter == 0:
                self.run_inference(current_iter)

        while current_iter < max_train_iters:
            for sample in self.dataloader_train:
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

                if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                    self.run_inference(current_iter)

                if current_iter >= max_train_iters:
                    break

        progress.close()

    def run_inference(self, current_iter):
        base_output_dir = self.infer_config.get("output_dir", "./output_infer")
        iter_output_dir = os.path.join(base_output_dir, f"iter-{current_iter:09d}")

        self.inferencer.output_infer_dir = iter_output_dir
        os.makedirs(iter_output_dir, exist_ok=True)
        self.inferencer.infer()

        self.model.set_lora_trainable()

    def save_checkpoint(self, iteration, save_total_limit):
        prune_checkpoints(self.output_train_dir, save_total_limit)

        save_dir = os.path.join(self.output_train_dir, f"checkpoint-{iteration:09d}")
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_lora_weights(save_dir)

        training_state = {
            "iteration": iteration,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
