import copy
import math
import os

import torch
from loguru import logger

from lightx2v_train.model_capabilities import DistributionMatchingCapability, LossResult, ParallelCapability
from lightx2v_train.model_zoo import build_loaded_model
from lightx2v_train.runtime.distributed import barrier, is_main_process
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.trainers.base import BaseTrainer
from lightx2v_train.trainers.flow_matching import FlowMatchingTrainer
from lightx2v_train.utils.constants import WAN_NEGATIVE_PROMPT
from lightx2v_train.utils.registry import TRAINER_REGISTER


@TRAINER_REGISTER("video_cfg_distill")
class VideoCfgDistillTrainer(FlowMatchingTrainer):
    """Distill teacher CFG velocity at the same noised real-video latent."""

    trainer_name = "video_cfg_distill"
    required_capabilities = (*BaseTrainer.required_capabilities, DistributionMatchingCapability)

    def _resolve_train_type(self):
        if "train_type" in self.training_config:
            raise ValueError("CFG distillation uses training.student.train_type; remove training.train_type.")
        student = self.training_config.get("student")
        if not isinstance(student, dict):
            raise ValueError("training.student must be a mapping.")
        train_type = student.get("train_type")
        if train_type not in {"full", "lora"}:
            raise ValueError("training.student.train_type must be 'full' or 'lora'.")
        return train_type

    def _get_lora_config(self):
        return self.training_config.get("student", {}).get("lora", {})

    def _get_optimizer_config(self):
        optimizer = self.training_config.get("student", {}).get("optimizer")
        if not isinstance(optimizer, dict):
            raise ValueError("training.student.optimizer must be a mapping.")
        return optimizer

    def __init__(self, config):
        super().__init__(config)
        cfg_distill = self.training_config.get("cfg_distill", {})
        teacher = self.training_config.get("teacher", {})
        if not isinstance(cfg_distill, dict):
            raise ValueError("training.cfg_distill must be a mapping.")
        if not isinstance(teacher, dict):
            raise ValueError("training.teacher must be a mapping.")
        self.guidance_scale = float(teacher.get("guidance_scale", 5.0))
        self.cfg_norm = str(teacher.get("cfg_norm", "none"))
        self.negative_prompt = str(cfg_distill.get("negative_prompt", WAN_NEGATIVE_PROMPT))
        if not math.isfinite(self.guidance_scale) or self.guidance_scale <= 1.0:
            raise ValueError("CFG distillation requires training.teacher.guidance_scale > 1.")
        if self.cfg_norm not in {"none", "scalar", "layer_norm"}:
            raise ValueError("training.teacher.cfg_norm must be one of 'none', 'scalar', or 'layer_norm'.")
        if config.get("data", {}).get("train", {}).get("name") != "video_dataset":
            raise ValueError("Real-data CFG distillation requires data.train.name=video_dataset.")

        # The reference samples this interval BEFORE time shifting, unlike
        # the shared scheduler's min_sigma/max_sigma post-shift clamping.
        self.min_t = float(config["scheduler"].get("min_t", 0.001))
        self.max_t = float(config["scheduler"].get("max_t", 1.0))
        if not 0 <= self.min_t < self.max_t <= 1:
            raise ValueError("CFG distillation requires 0 <= scheduler.min_t < scheduler.max_t <= 1.")
        if self.noise_scheduler.timestep_distribution not in {"uniform", "logitnormal"}:
            raise ValueError("Real-data CFG distillation supports uniform or logitnormal timestep sampling.")

    def set_model(self, model):
        BaseTrainer.set_model(self, model)
        self.student = model.capabilities.require(DistributionMatchingCapability)

    def setup(self, resume_ckpt_path=None):
        super().setup(resume_ckpt_path=resume_ckpt_path)
        teacher_config = copy.deepcopy(self.config)
        teacher_config["model"] = {key: copy.deepcopy(value) for key, value in self.model_config.items() if key not in {"fake", "teacher"}}
        teacher_override = self.model_config.get("teacher", {})
        if not isinstance(teacher_override, dict):
            raise ValueError("model.teacher must be a mapping.")
        teacher_config["model"]["transformer_param_dtype"] = self.model_config["running_dtype"]
        teacher_config["model"].update(copy.deepcopy(teacher_override))
        self.teacher_model = build_loaded_model(teacher_config, load_transformer=True, load_vae=False, load_condition_encoder=False)
        self.teacher_model.reuse_frozen_components_from(self.model)
        self.teacher = self.teacher_model.capabilities.require(DistributionMatchingCapability)
        self.teacher.denoiser().requires_grad_(False)
        self.teacher.set_training(False)
        self.teacher_model.capabilities.require(ParallelCapability).apply(self.config)
        self.teacher.set_training(False)
        logger.info(
            "[train] CFG distillation teacher model={} path={} guidance_scale={} cfg_norm={}",
            teacher_config["model"]["name"],
            teacher_config["model"]["pretrained_model_name_or_path"],
            self.guidance_scale,
            self.cfg_norm,
        )

    def _build_lr_scheduler(self, optimizer, num_training_steps=None, num_warmup_steps=None):
        min_lr = self.training_config.get("lr_min")
        if self.lr_scheduler_name != "cosine" or min_lr is None:
            return super()._build_lr_scheduler(optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
        min_lr = float(min_lr)
        total_steps = int(self.max_train_iters if num_training_steps is None else num_training_steps)
        warmup_steps = int(self.lr_warmup_iters if num_warmup_steps is None else num_warmup_steps)
        if not 0 <= warmup_steps < total_steps:
            raise ValueError("Cosine LR requires 0 <= lr_warmup_iters < num_training_steps.")
        lr_lambdas = []
        for group in optimizer.param_groups:
            peak_lr = float(group["lr"])
            if not 0 <= min_lr <= peak_lr or peak_lr <= 0:
                raise ValueError("training.lr_min must be between zero and the positive optimizer learning rate.")
            min_ratio = min_lr / peak_lr

            def lr_lambda(current_step, min_ratio=min_ratio):
                if current_step < warmup_steps:
                    return float(current_step) / max(1, warmup_steps)
                progress = min(1.0, float(current_step - warmup_steps) / max(1, total_steps - warmup_steps))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1.0 - min_ratio) * cosine

            lr_lambdas.append(lr_lambda)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)

    def _sample_sigma(self, latent):
        scheduler = self.noise_scheduler
        if scheduler.timestep_distribution == "uniform":
            sigma = torch.rand((1,), device=self.student.device, dtype=torch.float32)
        else:
            sigma = torch.randn((1,), device=self.student.device, dtype=torch.float32) * scheduler.logitnormal_std + scheduler.logitnormal_mean
            sigma = sigma.sigmoid()
        sigma = sigma * (self.max_t - self.min_t) + self.min_t
        sigma = scheduler.time_shift(sigma, latent_hw=self.student.latent_hw(latent.shape))
        return broadcast_sequence_parallel_value(sigma)

    def compute_loss_on_sample(self, sample):
        with torch.no_grad():
            latent = self.student.extract_real_latents(sample, self.model.latent_dtype, broadcast_sequence_parallel_value)
            noise = self.student.random_noise_like(latent, self.model.latent_dtype, broadcast_sequence_parallel_value)
            sigma = self._sample_sigma(latent)
            noisy_latent = self.student.add_noise(self.noise_scheduler, latent, noise, sigma)
            condition, negative = self.student.encode_conditions(sample, self.negative_prompt, self.guidance_scale, broadcast_sequence_parallel_value)
            target = self.teacher.predict_guided_velocity(noisy_latent, sigma, condition, negative, self.guidance_scale, self.cfg_norm)
        prediction = self.student.predict_velocity(noisy_latent, sigma, condition)
        loss = (prediction.float() - target.float()).square().mean()
        return LossResult(loss=loss, metrics={"sigma": sigma.float().mean(), "timestep": sigma.float().mean() * self.noise_scheduler.num_train_timesteps})

    def _run_teacher_inference(self, current_iter):
        output_dir = os.path.join(self.infer_config.get("output_dir", "./output_infer"), f"iter-{current_iter:09d}", "teacher")
        original_model = self.inferencer.model
        original_output_dir = self.inferencer.output_infer_dir
        original_negative_prompt = self.inferencer.negative_prompt
        keys = ("enable_cfg", "cfg_guidance_scale", "num_inference_steps", "denoising_step_list")
        original_settings = {key: self.inferencer.infer_config[key] for key in keys if key in self.inferencer.infer_config}
        try:
            self.inferencer.set_model(self.teacher_model)
            self.inferencer.output_infer_dir = output_dir
            self.inferencer.negative_prompt = self.negative_prompt
            self.inferencer.infer_config.update(enable_cfg=True, cfg_guidance_scale=self.guidance_scale, num_inference_steps=50)
            self.inferencer.infer_config.pop("denoising_step_list", None)
            if is_main_process():
                os.makedirs(output_dir, exist_ok=True)
                logger.info("[train] running teacher CFG inference iter={} output_dir={}", current_iter, output_dir)
            barrier()
            self.inferencer.infer()
            barrier()
        finally:
            self.inferencer.set_model(original_model)
            self.inferencer.output_infer_dir = original_output_dir
            self.inferencer.negative_prompt = original_negative_prompt
            for key in keys:
                self.inferencer.infer_config.pop(key, None)
            self.inferencer.infer_config.update(original_settings)
            self.teacher.set_training(False)
            self._restore_trainable_model(self.model)

    def run_inference(self, current_iter):
        if current_iter == 0:
            self._run_teacher_inference(current_iter)
        else:
            super().run_inference(current_iter)
