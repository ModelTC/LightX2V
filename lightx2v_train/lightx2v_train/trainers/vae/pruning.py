"""Task-aware layer search followed by export of an unadapted shallow decoder."""

import hashlib
import json
from pathlib import Path

import torch
from loguru import logger
from torch.distributed.tensor import DTensor

from lightx2v_train.runtime.distributed import barrier, is_main_process
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .distillation import VAEDistillationTrainer


@TRAINER_REGISTER("vae_pruning")
class VAEPruningTrainer(VAEDistillationTrainer):
    trainer_name = "vae_pruning"
    pruning_component_name = "decoder"

    def __init__(self, config):
        super().__init__(config)
        self.pruning_config = self.training_config["pruning"]
        self.ema_decay = float(self.pruning_config.get("ema_decay", 0.999))
        self.tau_start = float(self.pruning_config.get("tau_start", 4.0))
        self.tau_end = float(self.pruning_config.get("tau_end", 0.1))
        if not 0 <= self.ema_decay < 1 or min(self.tau_start, self.tau_end) <= 0:
            raise ValueError("Pruning requires 0 <= ema_decay < 1 and positive Gumbel temperatures.")
        if self.training_config.get("vae_distillation", {}).get("gan", {}).get("enabled", False):
            raise ValueError("Layer search uses reconstruction and perceptual loss; enable GAN during recovery.")
        schedule = {
            "pruning": self.pruning_config,
            self.pruning_component_name: self.model_config[f"pruned_{self.pruning_component_name}"],
            "max_train_iters": self.max_train_iters,
        }
        self.pruning_fingerprint = hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()

    def _pruning_component(self):
        return getattr(self.model.denoiser_module(), self.pruning_component_name)

    def _gate_values(self):
        gate = self._pruning_component().gate_logits.detach()
        return (gate.full_tensor() if isinstance(gate, DTensor) else gate).float()

    def _build_optimizer(self, params, optimizer_config=None):
        gate = self._pruning_component().gate_logits
        multiplier = float(self.pruning_config.get("gate_lr_multiplier", 10.0))
        groups = [
            {"params": [param for param in params if param is not gate]},
            {"params": [gate], "lr": self.optimizer_learning_rate * multiplier, "weight_decay": 0.0},
        ]
        return super()._build_optimizer(groups, optimizer_config)

    def setup(self, resume_ckpt_path=None):
        if self.model.search_config is None:
            raise ValueError(f"{self.trainer_name} requires model.pruned_{self.pruning_component_name}.search.")
        super().setup(resume_ckpt_path=None)
        self.gate_ema = self._gate_values().clone()
        self.gate_ema_iteration = self.current_train_iteration
        if resume_ckpt_path is not None:
            self._load_resume_state(resume_ckpt_path)

    @torch.no_grad()
    def _update_gate_ema(self):
        # Called only after the preceding optimizer update, never once per micro-batch.
        if self.gate_ema_iteration < self.current_train_iteration:
            self.gate_ema.lerp_(self._gate_values(), 1.0 - self.ema_decay)
            self.gate_ema_iteration = self.current_train_iteration

    def compute_loss_on_sample(self, sample):
        self._update_gate_ema()
        progress = self.current_train_iteration / max(1, self.max_train_iters - 1)
        temperature = self.tau_start + (self.tau_end - self.tau_start) * progress
        student = self.model.denoiser_module()
        self._pruning_component().temperature = temperature
        student.prepare_search_step(sample["inputs"]["video"].shape[0])
        result = super().compute_loss_on_sample(sample)
        result.metrics["pruning_temperature"] = temperature
        return result

    def save_checkpoint(self, iteration, save_total_limit):
        self._update_gate_ema()
        super().save_checkpoint(iteration, save_total_limit)
        logger.info("[prune] iter={} EMA retained layers={}", iteration, self.model.denoiser_module().selected_layers(logits=self.gate_ema))

    def _extra_checkpoint_state(self):
        return {
            **super()._extra_checkpoint_state(),
            "pruning_schedule": self.pruning_fingerprint,
            "gate_ema": self.gate_ema.cpu(),
            "gate_ema_iteration": self.gate_ema_iteration,
        }

    def _load_extra_checkpoint_state(self, state):
        super()._load_extra_checkpoint_state(state)
        if state.get("pruning_schedule") != self.pruning_fingerprint:
            raise ValueError("Pruning schedule differs from the resumed checkpoint.")
        self.gate_ema.copy_(state["gate_ema"])
        self.gate_ema_iteration = int(state["gate_ema_iteration"])

    def train(self):
        super().train()
        self._update_gate_ema()
        if not self.save_every_iters or self.current_train_iteration % self.save_every_iters:
            self.save_checkpoint(self.current_train_iteration, self.save_total_limit)
        # All ranks finish their gate collective before rank zero does CPU-only export.
        barrier()
        if is_main_process():
            output_dir = self.pruning_config.get("export_dir", str(Path(self.output_train_dir) / "export"))
            self._export_pruned_model(output_dir)
        barrier()

    def _export_pruned_model(self, output_dir):
        self.model.export_pruned_decoder(output_dir, self.gate_ema.cpu())
