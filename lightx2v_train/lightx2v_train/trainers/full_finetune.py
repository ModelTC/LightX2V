from __future__ import annotations

import os

import torch
from diffusers.optimization import get_scheduler

from lightx2v_train.utils.registry import TRAINER_REGISTER

from .lora import LoraTrainer


@TRAINER_REGISTER("full_finetune")
class FullFinetuneTrainer(LoraTrainer):
    def setup(self) -> None:
        self.model.set_full_trainable()
        if self.config.get("gradient_checkpointing", True):
            self.model.enable_gradient_checkpointing()

        optim_config = self.config.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=optim_config.get("learning_rate", 1e-5),
            betas=(optim_config.get("adam_beta1", 0.9), optim_config.get("adam_beta2", 0.999)),
            weight_decay=optim_config.get("weight_decay", 0.01),
            eps=optim_config.get("adam_epsilon", 1e-8),
        )
        self.lr_scheduler = get_scheduler(
            self.config.get("lr_scheduler", "constant"),
            optimizer=self.optimizer,
            num_warmup_steps=self.config.get("lr_warmup_steps", 0),
            num_training_steps=self.config["max_train_steps"],
        )

    def save_checkpoint(self, step: int, checkpoint_limit: int | None) -> None:
        output_dir = self.config["output_dir"]
        save_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_full_model(save_dir)
