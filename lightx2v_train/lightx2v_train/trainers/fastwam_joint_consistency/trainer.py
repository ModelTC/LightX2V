import os
import time
from copy import deepcopy

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from loguru import logger
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.model_zoo.native.wan.fastwam.action_distill import (
    CachedActionDenoiser,
    build_action_distill_condition,
    sample_action_one_step,
    sample_action_teacher,
)
from lightx2v_train.runtime.distributed import (
    barrier,
    get_data_parallel_group,
    get_world_size,
    is_distributed,
    is_main_process,
    is_sequence_parallel_enabled,
    reduce_mean,
)
from lightx2v_train.runtime.monitor import build_monitor
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .checkpoint import JointConsistencyCheckpointManager
from .config import FastWAMJointConsistencyConfig
from .roles import ConsistencyRoles, JointConsistencyDenoiser


def shifted_consistency_pair(base_sigma, shift, target_steps):
    """Apply FastWAM's shift to a pair separated by one distilled step."""
    sigma_end = (base_sigma - 1.0 / target_steps).clamp(min=0.0)

    def shift_sigma(value):
        return shift * value / (1.0 + (shift - 1.0) * value)

    return shift_sigma(base_sigma), shift_sigma(sigma_end)


def _expand_sigma(sigma, value):
    return sigma.reshape(sigma.shape[0], *([1] * (value.ndim - 1)))


def _masked_mean(error, valid_mask):
    if valid_mask is None:
        return error.mean()
    mask = valid_mask.to(device=error.device, dtype=error.dtype).unsqueeze(-1).expand_as(error)
    return (error * mask).sum() / mask.sum().clamp(min=1.0)


def _masked_pseudo_huber(prediction, target, valid_mask, c):
    difference = prediction.float() - target.float()
    return _masked_mean(torch.sqrt(difference.square() + c**2) - c, valid_mask)


def _masked_mse(prediction, target, valid_mask):
    return _masked_mean(F.mse_loss(prediction.float(), target.float(), reduction="none"), valid_mask)


def _masked_l1_per_sample(prediction, target, valid_mask):
    error = (prediction.float() - target.float()).abs()
    reduce_dims = tuple(range(1, error.ndim))
    if valid_mask is None:
        return error.mean(dim=reduce_dims)
    mask = valid_mask.to(device=error.device, dtype=error.dtype).unsqueeze(-1).expand_as(error)
    return (error * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp(min=1.0)


def _slice_batch(value, size):
    if isinstance(value, torch.Tensor):
        return value[:size]
    if isinstance(value, dict):
        return {key: _slice_batch(item, size) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return value[:size]
    return value


@TRAINER_REGISTER("fastwam_joint_consistency")
class FastWAMJointConsistencyTrainer:
    def __init__(self, config):
        self.config = config
        self.runtime_config = deepcopy(config)
        self.training_config = config["training"]
        self.inference_config = config.get("inference", {})
        self.logging_config = config.get("logging", {})
        self.parsed = FastWAMJointConsistencyConfig.from_mapping(config)
        self.output_dir = self.training_config["output_dir"]
        self.max_train_iters = int(self.training_config["max_train_iters"])
        self.gradient_accumulation_iters = max(1, int(self.training_config.get("gradient_accumulation_iters", 1)))
        self.max_grad_norm = float(self.training_config.get("max_grad_norm", 1.0))
        self.save_every_iters = int(self.training_config.get("save_every_iters", 0) or 0)
        self.save_total_limit = int(self.training_config.get("save_total_limit", 3))
        self.save_final = bool(self.training_config.get("save_final", True))
        self.log_every_iters = max(1, int(self.logging_config.get("train_log_every_iters", 10)))
        self.eval_every_iters = int(self.inference_config.get("infer_every_iters", 0) or 0)
        self.eval_num_samples = max(1, int(self.inference_config.get("num_samples", 1)))
        self.eval_seed = int(self.inference_config.get("seed", 42))
        self.checkpoints = JointConsistencyCheckpointManager(self)
        if is_main_process():
            os.makedirs(self.output_dir, exist_ok=True)
        self.monitor = build_monitor(config)

    def set_model(self, model):
        self.model = model

    def set_data(self, dataloader_train, dataloader_eval=None):
        self.dataloader_train = dataloader_train
        self.dataloader_eval = dataloader_eval

    def setup(self):
        sequence_parallel = self.config.get("distributed", {}).get("sequence_parallel", {})
        sequence_parallel_enabled = sequence_parallel.get("enabled", False) if isinstance(sequence_parallel, dict) else bool(sequence_parallel)
        if sequence_parallel_enabled or is_sequence_parallel_enabled():
            raise ValueError("fastwam_joint_consistency does not support sequence parallelism.")

        module = self.model.unwrap_module()
        module.eval().requires_grad_(False)
        self.roles = ConsistencyRoles.build(module.action_expert, self.parsed.student)
        self.video_roles = ConsistencyRoles.build(module.video_expert, self.parsed.student) if self.parsed.train_video else None
        for name in ("student", "target", "teacher"):
            expert = getattr(self.roles, name)
            denoiser = JointConsistencyDenoiser(expert, getattr(self.video_roles, name), module) if self.video_roles is not None else CachedActionDenoiser(expert, module.mot)
            if name != "student":
                denoiser.eval()
            setattr(self, f"{name}_denoiser", denoiser)
        if self.training_config.get("gradient_checkpointing", False):
            self.student_denoiser.action_module().use_gradient_checkpointing = True
            if self.video_roles is not None:
                video = self.video_roles.student
                video = video.get_base_model() if hasattr(video, "get_base_model") else video
                video.use_gradient_checkpointing = True
                self.student_denoiser.mot.mot_checkpoint_mixed_attn = True

        self.student_params = self.roles.trainable_parameters
        if self.video_roles is not None:
            self.student_params += self.video_roles.trainable_parameters
        if not self.student_params:
            raise RuntimeError("FastWAM joint consistency has no trainable student parameters.")
        optimizer_config = self.parsed.student.optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_params,
            lr=float(optimizer_config.get("learning_rate", 1e-4)),
            betas=(float(optimizer_config.get("adam_beta1", 0.9)), float(optimizer_config.get("adam_beta2", 0.95))),
            weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
            eps=float(optimizer_config.get("adam_epsilon", 1e-8)),
        )
        self.scheduler = get_scheduler(
            self.training_config.get("lr_scheduler", "constant"),
            optimizer=self.optimizer,
            num_warmup_steps=int(self.training_config.get("lr_warmup_iters", 0)),
            num_training_steps=self.max_train_iters,
        )
        if is_distributed():
            self.student_denoiser = DistributedDataParallel(
                self.student_denoiser,
                device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
                process_group=get_data_parallel_group(),
                find_unused_parameters=self.parsed.train_video,
            )
        # DDP broadcasts the online student; initialize every rank's EMA from it.
        self.roles.copy_student_to_target()
        if self.video_roles is not None:
            self.video_roles.copy_student_to_target()

        resume_path, current_iter = self.checkpoints.resolve_resume()
        if resume_path is not None:
            current_iter = self.checkpoints.load(resume_path)
            logger.info("[resume] restored FastWAM joint consistency from {} at iteration {}", resume_path, current_iter)
        return current_iter

    def _prepare_batch(self, sample):
        module = self.model.unwrap_module()
        with torch.no_grad(), self.model.autocast_context():
            inputs = module.build_action_distill_inputs(sample)
            # Joint denoisers build their own cache inside each role's forward.
            condition = inputs if self.parsed.train_video else build_action_distill_condition(module, inputs)
        valid_mask = None if inputs["action_is_pad"] is None else ~inputs["action_is_pad"]
        return inputs, condition, valid_mask

    def _sigma_pair(self, action):
        scheduler = self.model.unwrap_module().train_action_scheduler
        base_sigma = torch.rand(action.shape[0], device=action.device, dtype=torch.float32)
        sigma_start, sigma_end = shifted_consistency_pair(base_sigma, scheduler.shift, self.parsed.target_steps)
        return sigma_start.to(action.dtype), sigma_end.to(action.dtype)

    def _loss(self, inputs, condition, valid_mask):
        action = inputs["action"]
        noise = torch.randn_like(action)
        sigma_start, sigma_end = self._sigma_pair(action)
        sigma_start_expanded = _expand_sigma(sigma_start, action)
        sigma_end_expanded = _expand_sigma(sigma_end, action)
        noisy_action = (1.0 - sigma_start_expanded) * action + sigma_start_expanded * noise
        num_timesteps = float(self.model.unwrap_module().train_action_scheduler.num_train_timesteps)
        timestep_start = sigma_start * num_timesteps
        timestep_end = sigma_end * num_timesteps

        with torch.no_grad():
            teacher_velocity = self.teacher_denoiser(noisy_action, timestep_start, condition)
            endpoint_action = noisy_action + (sigma_end_expanded - sigma_start_expanded) * teacher_velocity

        student_velocity = self.student_denoiser(noisy_action, timestep_start, condition)
        student_x0 = noisy_action - sigma_start_expanded * student_velocity
        with torch.no_grad():
            target_velocity = self.target_denoiser(endpoint_action, timestep_end, condition)
            target_x0 = endpoint_action - sigma_end_expanded * target_velocity

        consistency_loss = _masked_pseudo_huber(student_x0, target_x0, valid_mask, self.parsed.huber_c)
        flow_loss = _masked_mse(student_velocity, noise - action, valid_mask)
        loss = self.parsed.consistency_loss_weight * consistency_loss + self.parsed.flow_loss_weight * flow_loss
        return loss, {"consistency": consistency_loss.detach(), "flow": flow_loss.detach()}

    def _iter_train_samples(self):
        epoch = 0
        while True:
            sampler = getattr(self.dataloader_train, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            yield from self.dataloader_train
            epoch += 1

    @torch.no_grad()
    def evaluate(self, current_iter):
        if self.dataloader_eval is None:
            return
        totals = {"ema_teacher_l1": 0.0, "ema_gt_l1": 0.0}
        count = 0
        generator = torch.Generator(device=self.model.unwrap_module().device).manual_seed(self.eval_seed)
        for sample in self.dataloader_eval:
            batch_size = int(sample["video"].shape[0])
            remaining = self.eval_num_samples - count
            if batch_size > remaining:
                sample = _slice_batch(sample, remaining)
                batch_size = remaining
            inputs, condition, valid_mask = self._prepare_batch(sample)
            noise = torch.randn(inputs["action"].shape, generator=generator, device=inputs["action"].device, dtype=inputs["action"].dtype)
            module = self.model.unwrap_module()
            with self.model.autocast_context():
                target_condition = self.target_denoiser.build_condition(inputs) if self.parsed.train_video else condition
                teacher_condition = self.teacher_denoiser.build_condition(inputs) if self.parsed.train_video else condition
                ema_action = sample_action_one_step(self.target_denoiser, noise, target_condition, module.train_action_scheduler.num_train_timesteps)
                teacher_action = sample_action_teacher(self.teacher_denoiser, noise, teacher_condition, module.infer_action_scheduler, self.parsed.teacher_reference_steps)
            totals["ema_teacher_l1"] += float(_masked_l1_per_sample(ema_action, teacher_action, valid_mask).sum().item())
            totals["ema_gt_l1"] += float(_masked_l1_per_sample(ema_action, inputs["action"], valid_mask).sum().item())
            count += batch_size
            if count >= self.eval_num_samples:
                break
        metrics = {f"eval/{name}": reduce_mean(total / count) for name, total in totals.items()} if count else {}
        if count and is_main_process():
            logger.info("[eval] iter={} ema_teacher_l1={:.6f} ema_gt_l1={:.6f}", current_iter, metrics["eval/ema_teacher_l1"], metrics["eval/ema_gt_l1"])
            self.monitor.log_metrics(metrics, step=current_iter)

    def train(self):
        current_iter = self.setup()
        start_iter = current_iter
        barrier()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        samples = self._iter_train_samples()
        started_at = time.perf_counter()
        logger.info(
            "[train] start method=fastwam_joint_consistency iter={}/{} world_size={} global_batch={} target_steps={} ema_decay={} train_video={}",
            current_iter,
            self.max_train_iters,
            get_world_size(),
            int(self.config["data"]["train"]["batch_size"]) * get_world_size() * self.gradient_accumulation_iters,
            self.parsed.target_steps,
            self.parsed.ema_decay,
            self.parsed.train_video,
        )
        while current_iter < self.max_train_iters:
            self.optimizer.zero_grad(set_to_none=True)
            accumulated = {"consistency": 0.0, "flow": 0.0}
            for _ in range(self.gradient_accumulation_iters):
                inputs, condition, valid_mask = self._prepare_batch(next(samples))
                with self.model.autocast_context():
                    loss, metrics = self._loss(inputs, condition, valid_mask)
                (loss / self.gradient_accumulation_iters).backward()
                for name in accumulated:
                    accumulated[name] += float(metrics[name].item()) / self.gradient_accumulation_iters

            grad_norm = torch.nn.utils.clip_grad_norm_(self.student_params, self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.roles.update_target(self.parsed.ema_decay)
            if self.video_roles is not None:
                self.video_roles.update_target(self.parsed.ema_decay)
            current_iter += 1

            if current_iter == 1 or current_iter % self.log_every_iters == 0:
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                metrics = {
                    "train/consistency_loss": reduce_mean(accumulated["consistency"]),
                    "train/flow_loss": reduce_mean(accumulated["flow"]),
                    "train/grad_norm": reduce_mean(float(grad_norm)),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/iters_per_second": (current_iter - start_iter) / elapsed,
                }
                if torch.cuda.is_available():
                    metrics["system/gpu_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(
                    "[train] iter={}/{} consistency={:.6f} flow={:.6f} grad={:.4f} speed={:.3f} it/s max_mem={:.2f}GiB",
                    current_iter,
                    self.max_train_iters,
                    metrics["train/consistency_loss"],
                    metrics["train/flow_loss"],
                    metrics["train/grad_norm"],
                    metrics["train/iters_per_second"],
                    metrics.get("system/gpu_max_memory_allocated_gib", 0.0),
                )
                self.monitor.log_metrics(metrics, step=current_iter)
            if self.eval_every_iters and current_iter % self.eval_every_iters == 0:
                self.evaluate(current_iter)
            if self.save_every_iters and current_iter % self.save_every_iters == 0:
                self.checkpoints.save(current_iter)

        if self.save_final and (not self.save_every_iters or current_iter % self.save_every_iters):
            self.checkpoints.save(current_iter)
        logger.info("[train] finished FastWAM joint consistency iter={}", current_iter)
        self.monitor.finish()
