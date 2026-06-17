import os
import shutil

import torch
from diffusers.optimization import get_scheduler
from loguru import logger

from lightx2v_train.infer import build_inferencer
from lightx2v_train.runtime.checkpoint import parse_checkpoint_iteration, prune_checkpoints
from lightx2v_train.runtime.distributed import barrier, get_world_size, is_main_process
from lightx2v_train.runtime.fsdp import apply_fsdp2
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .lora import LoraTrainer


@TRAINER_REGISTER("full")
class FullTrainer(LoraTrainer):
    def setup(self, resume_ckpt_path=None):
        if resume_ckpt_path is not None and get_world_size() == 1:
            self.model.load_full_weights_for_resume(resume_ckpt_path)

        self.model.set_full_trainable()

        apply_fsdp2(self.model, self.config)

        if self.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        if self.infer_every_iters:
            self.inferencer = build_inferencer(self.config)
            self.inferencer.set_model(self.model)

        self.model.log_model_structure()

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
            if self.model.is_fsdp2_wrapped():
                self._load_distributed_state(resume_ckpt_path)
            else:
                self._load_single_process_training_state(resume_ckpt_path)

    def _load_single_process_training_state(self, resume_ckpt_path):
        training_state_path = os.path.join(resume_ckpt_path, "training_state.pt")
        if not os.path.exists(training_state_path):
            raise RuntimeError(f"training_state.pt not found in {resume_ckpt_path}")

        state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_metadata(state, training_state_path, resume_ckpt_path)
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        logger.info("Restored full training state from {}", training_state_path)

    def run_inference(self, current_iter):
        base_output_dir = self.infer_config.get("output_dir", "./output_infer")
        iter_output_dir = os.path.join(base_output_dir, f"iter-{current_iter:09d}")

        self.inferencer.output_infer_dir = iter_output_dir
        os.makedirs(iter_output_dir, exist_ok=True)
        logger.info("[train] running inference iter={} output_dir={}", current_iter, iter_output_dir)
        self.inferencer.infer()
        barrier()
        logger.info("[train] finished inference iter={}", current_iter)

        self.model.set_full_trainable()

    def _is_fsdp2_checkpoint_expected(self):
        fsdp_config = self.config.get("distributed", {}).get("fsdp2", {})
        return get_world_size() > 1 and fsdp_config.get("enabled", True)

    def _is_complete_checkpoint(self, ckpt_path):
        if self._is_fsdp2_checkpoint_expected():
            return os.path.isdir(os.path.join(ckpt_path, "dist_state")) and os.path.exists(os.path.join(ckpt_path, "trainer_state.pt"))

        return os.path.isdir(os.path.join(ckpt_path, "transformer")) and os.path.exists(os.path.join(ckpt_path, "training_state.pt"))

    def _resolve_resume(self):
        if not self.auto_resume:
            return None, 0
        if not os.path.exists(self.output_train_dir):
            logger.info("Auto-resume enabled but no checkpoint found in '{}'. Starting from scratch.", self.output_train_dir)
            return None, 0

        checkpoints = [name for name in os.listdir(self.output_train_dir) if name.startswith("checkpoint-")]
        checkpoints = sorted(checkpoints, key=parse_checkpoint_iteration, reverse=True)
        for name in checkpoints:
            ckpt_path = os.path.join(self.output_train_dir, name)
            if self._is_complete_checkpoint(ckpt_path):
                current_iter = parse_checkpoint_iteration(ckpt_path)
                logger.info("Auto-resuming from checkpoint: {} (iteration {})", ckpt_path, current_iter)
                return ckpt_path, current_iter
            logger.warning("Skipping incomplete checkpoint during auto-resume: {}", ckpt_path)

        logger.info("Auto-resume enabled but no complete checkpoint found in '{}'. Starting from scratch.", self.output_train_dir)
        return None, 0

    def save_checkpoint(self, iteration, save_total_limit):
        if is_main_process():
            prune_checkpoints(self.output_train_dir, save_total_limit)

        save_dir = os.path.join(self.output_train_dir, f"checkpoint-{iteration:09d}")
        logger.info("[train] saving full checkpoint iter={} path={}", iteration, save_dir)
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
        barrier()

        config_path = self.config.get("config_path")
        if is_main_process() and config_path is not None:
            shutil.copy2(config_path, os.path.join(save_dir, "config.yaml"))
        barrier()

        if self.model.is_fsdp2_wrapped():
            self._save_distributed_state(save_dir, iteration)
            barrier()
            logger.info("[train] saved full distributed checkpoint iter={} path={}", iteration, save_dir)
            return

        self.model.save_full_model(save_dir)
        barrier()

        training_state = {
            "iteration": iteration,
            "world_size": get_world_size(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        if is_main_process():
            torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
        barrier()
        logger.info("[train] saved full checkpoint iter={} path={}", iteration, save_dir)
