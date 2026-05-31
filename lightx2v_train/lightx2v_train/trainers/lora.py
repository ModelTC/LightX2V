import os
import shutil

import torch
import torch.distributed.checkpoint as dcp
from diffusers.optimization import get_scheduler
from loguru import logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from lightx2v_train.infer import build_inferencer
from lightx2v_train.runtime.checkpoint import find_latest_checkpoint, parse_checkpoint_iteration, prune_checkpoints
from lightx2v_train.runtime.distributed import barrier, get_world_size, is_main_process, reduce_mean
from lightx2v_train.runtime.fsdp import apply_fsdp2
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
        logging_config = self.config.get("logging", {})
        self.train_log_every_iters = max(1, int(logging_config.get("train_log_every_iters", 10)))

        resume_config = self.config.get("resume", {})
        self.auto_resume = resume_config.get("auto_resume", False)

    def setup(self, resume_ckpt_path=None):
        self.model.add_lora(self.lora_rank, self.lora_alpha, self.lora_target_modules)
        self.model.set_lora_trainable()

        apply_fsdp2(self.model, self.config)

        if self.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        if self.infer_every_iters:
            self.inferencer = build_inferencer(self.config)
            self.inferencer.set_model(self.model)

        self.trainable_params = list(self.model.trainable_parameters())
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
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
            self._load_resume_state(resume_ckpt_path)

    def _load_resume_state(self, resume_ckpt_path):
        if self.model.is_fsdp2_wrapped():
            self._load_distributed_state(resume_ckpt_path)
            return

        self._load_single_process_state(resume_ckpt_path)

    def _load_single_process_state(self, resume_ckpt_path):
        training_state_path = os.path.join(resume_ckpt_path, "training_state.pt")
        if not os.path.exists(training_state_path):
            raise RuntimeError(f"training_state.pt not found in {resume_ckpt_path}")

        state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_metadata(state, training_state_path, resume_ckpt_path)
        self.model.load_lora_weights_for_resume(resume_ckpt_path)
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        logger.info("Restored training state from {}", training_state_path)

    def _load_distributed_state(self, resume_ckpt_path):
        dist_state_path = os.path.join(resume_ckpt_path, "dist_state")
        if not os.path.exists(dist_state_path):
            raise RuntimeError(f"FSDP2 resume requires dist_state/, but it was not found in {resume_ckpt_path}")

        trainer_state_path = os.path.join(resume_ckpt_path, "trainer_state.pt")
        if not os.path.exists(trainer_state_path):
            raise RuntimeError(f"trainer_state.pt not found in {resume_ckpt_path}")
        trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_metadata(trainer_state, trainer_state_path, resume_ckpt_path)

        options = StateDictOptions(ignore_frozen_params=True, strict=False)
        state_module = self.model.fsdp2_state_module()
        model_state, optim_state = get_state_dict(state_module, self.optimizer, options=options)
        state = {"model": model_state, "optimizer": optim_state}
        dcp.load(state, checkpoint_id=dist_state_path)
        set_state_dict(
            state_module,
            self.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
            options=options,
        )

        self.lr_scheduler.load_state_dict(trainer_state["lr_scheduler"])
        logger.info("Restored distributed training state from {}", resume_ckpt_path)

    def _validate_checkpoint_metadata(self, state, state_path, resume_ckpt_path):
        checkpoint_world_size = state.get("world_size")
        current_world_size = get_world_size()
        if checkpoint_world_size != current_world_size:
            raise RuntimeError(f"Cannot resume checkpoint saved with world_size={checkpoint_world_size} using world_size={current_world_size}: {state_path}")

        expected_iteration = parse_checkpoint_iteration(resume_ckpt_path)
        checkpoint_iteration = state.get("iteration")
        if checkpoint_iteration != expected_iteration:
            raise RuntimeError(f"Cannot resume checkpoint with iteration={checkpoint_iteration} in {state_path}, expected iteration={expected_iteration} from {resume_ckpt_path}")

    def compute_loss_on_sample(self, sample):
        with torch.no_grad():
            latent = self.model.encode_to_latent(sample)
            n = latent.shape[0]
            noise = torch.randn_like(latent, dtype=self.running_dtype)
            latent_hw = (latent.shape[3], latent.shape[4])
            timestep_or_sigma = self.noise_scheduler.sample_timestep_or_sigma(n, latent_hw=latent_hw)
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
            logger.info("Auto-resume enabled but no checkpoint found in '{}'. Starting from scratch.", self.output_train_dir)
        else:
            logger.info("Auto-resuming from checkpoint: {} (iteration {})", ckpt_path, current_iter)
        return ckpt_path, current_iter

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
            "[train] start iter={}/{} world_size={} grad_accum={} train_log_every_iters={}",
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

    def _set_gradient_sync(self, enabled):
        self.model.set_fsdp2_gradient_sync(enabled)

    def run_inference(self, current_iter):
        base_output_dir = self.infer_config.get("output_dir", "./output_infer")
        iter_output_dir = os.path.join(base_output_dir, f"iter-{current_iter:09d}")

        self.inferencer.output_infer_dir = iter_output_dir
        os.makedirs(iter_output_dir, exist_ok=True)
        logger.info("[train] running inference iter={} output_dir={}", current_iter, iter_output_dir)
        self.inferencer.infer()
        barrier()
        logger.info("[train] finished inference iter={}", current_iter)

        self.model.set_lora_trainable()

    def save_checkpoint(self, iteration, save_total_limit):
        if is_main_process():
            prune_checkpoints(self.output_train_dir, save_total_limit)

        save_dir = os.path.join(self.output_train_dir, f"checkpoint-{iteration:09d}")
        logger.info("[train] saving checkpoint iter={} path={}", iteration, save_dir)
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
        barrier()

        self.model.save_lora_weights(save_dir)
        barrier()

        config_path = self.config.get("config_path")
        if is_main_process() and config_path is not None:
            shutil.copy2(config_path, os.path.join(save_dir, "config.yaml"))

        if self.model.is_fsdp2_wrapped():
            self._save_distributed_state(save_dir, iteration)
            barrier()
            logger.info("[train] saved checkpoint iter={} path={}", iteration, save_dir)
            return

        training_state = {
            "iteration": iteration,
            "world_size": get_world_size(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        if is_main_process():
            torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
        barrier()
        logger.info("[train] saved checkpoint iter={} path={}", iteration, save_dir)

    def _save_distributed_state(self, save_dir, iteration):
        dist_state_path = os.path.join(save_dir, "dist_state")
        if is_main_process():
            os.makedirs(dist_state_path, exist_ok=True)
            torch.save(
                {
                    "iteration": iteration,
                    "world_size": get_world_size(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                },
                os.path.join(save_dir, "trainer_state.pt"),
            )
        barrier()

        options = StateDictOptions(ignore_frozen_params=True, strict=False)
        model_state, optim_state = get_state_dict(self.model.fsdp2_state_module(), self.optimizer, options=options)
        dcp.save(
            {"model": model_state, "optimizer": optim_state},
            checkpoint_id=dist_state_path,
        )
