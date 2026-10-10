import math

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_train.model_capabilities import LossResult
from lightx2v_train.runtime.distributed import is_distributed
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.schedulers.wan_unipc import build_wan_unipc_scheduler, wan_unipc_timestep_to_sigma
from lightx2v_train.trainers.cfg_distill.trainer import VideoCfgDistillTrainer
from lightx2v_train.utils.registry import TRAINER_REGISTER


@TRAINER_REGISTER("video_cfg_trajectory_distill")
class VideoCfgTrajectoryDistillTrainer(VideoCfgDistillTrainer):
    """Distill teacher CFG targets sampled from cached UniPC trajectories."""

    trainer_name = "video_cfg_trajectory_distill"

    def _valid_train_data_names(self):
        return super()._valid_train_data_names() | {"prompt_dataset"}

    def __init__(self, config):
        super().__init__(config)
        trajectory_config = self.training_config.get("cfg_trajectory_distill", {})
        if not isinstance(trajectory_config, dict):
            raise ValueError("training.cfg_trajectory_distill must be a mapping.")
        self.trajectory_refresh_iters = int(trajectory_config.get("refresh_every_iters", 10))
        self.trajectory_rollout_steps = int(trajectory_config.get("rollout_steps", 50))
        self.trajectory_intervals = int(trajectory_config.get("intervals", 10))
        self.unipc_shift = float(trajectory_config.get("unipc_shift", 5.0))
        self.rollout_policy = str(trajectory_config.get("rollout_policy", "teacher"))
        uncond_config = trajectory_config.get("uncond", {})
        if not isinstance(uncond_config, dict):
            raise ValueError("training.cfg_trajectory_distill.uncond must be a mapping.")
        self.uncond_enabled = bool(uncond_config.get("enabled", False))
        self.uncond_loss_weight = float(uncond_config.get("loss_weight", 0.5))
        turn_config = trajectory_config.get("turn_loss", {})
        if not isinstance(turn_config, dict) or turn_config.get("enabled", False):
            raise ValueError("CFG trajectory distillation does not support turn_loss; leave it disabled.")

        if self.gradient_accumulation_iters != 1:
            raise ValueError("CFG trajectory distillation currently requires training.gradient_accumulation_iters=1.")
        if self.trajectory_intervals <= 0:
            raise ValueError("intervals must be positive.")
        if self.trajectory_refresh_iters != self.trajectory_intervals:
            raise ValueError("refresh_every_iters must equal intervals so every cached trajectory point is consumed exactly once.")
        if self.trajectory_rollout_steps < self.trajectory_intervals or self.trajectory_rollout_steps % self.trajectory_intervals != 0:
            raise ValueError("rollout_steps must be divisible by intervals.")
        if not math.isfinite(self.unipc_shift) or self.unipc_shift <= 0:
            raise ValueError("unipc_shift must be finite and positive.")
        if self.rollout_policy not in {"teacher", "student"}:
            raise ValueError("training.cfg_trajectory_distill.rollout_policy must be 'teacher' or 'student'.")
        if not math.isfinite(self.uncond_loss_weight) or self.uncond_loss_weight < 0:
            raise ValueError("uncond.loss_weight must be finite and non-negative.")

        self._trajectory_cache = None
        self._trajectory_cursor = 0

    def _build_unipc_scheduler(self):
        return build_wan_unipc_scheduler(
            self.noise_scheduler.num_train_timesteps,
            self.trajectory_rollout_steps,
            device=self.model.device,
            shift=self.unipc_shift,
        )

    @staticmethod
    def _sample_stratified_steps(batch_size, rollout_steps, intervals, device):
        steps_per_interval = rollout_steps // intervals
        offsets = torch.randint(0, steps_per_interval, (batch_size, intervals), device=device)
        starts = torch.arange(intervals, device=device) * steps_per_interval
        return offsets + starts.unsqueeze(0)

    @staticmethod
    def _sample_without_replacement_orders(batch_size, intervals, device):
        return torch.stack([torch.randperm(intervals, device=device) for _ in range(batch_size)], dim=0)

    @staticmethod
    def _broadcast_across_ranks(tensor):
        """Keep conditional teacher forwards in the same order on all FSDP ranks."""
        if is_distributed():
            dist.broadcast(tensor, src=0)
        return tensor

    def _initial_rollout_latents(self):
        train_data_config = self.config["data"]["train"]
        latent = self.model.prepare_infer_latents(int(train_data_config["height"]), int(train_data_config["width"]))
        if latent.shape[0] != 1:
            raise ValueError("CFG trajectory distillation requires physical batch size 1.")
        return broadcast_sequence_parallel_value(latent.to(device=self.model.device, dtype=self.model.latent_dtype))

    @torch.no_grad()
    def _refresh_trajectory_cache(self, sample):
        condition, negative_condition = self.student.encode_conditions(sample, self.negative_prompt, self.guidance_scale, broadcast_sequence_parallel_value)
        latent = self._initial_rollout_latents()
        batch_size = int(latent.shape[0])
        scheduler = self._build_unipc_scheduler()
        selected_steps = self._sample_stratified_steps(batch_size, self.trajectory_rollout_steps, self.trajectory_intervals, latent.device)
        selected_steps = broadcast_sequence_parallel_value(selected_steps)
        student_rollout = self.rollout_policy == "student"
        if student_rollout:
            # Teacher and student are separate FSDP models. Every rank must
            # interleave their forwards identically, even with different prompts.
            selected_steps = self._broadcast_across_ranks(selected_steps)
        consume_orders = self._sample_without_replacement_orders(batch_size, self.trajectory_intervals, latent.device)
        consume_orders = broadcast_sequence_parallel_value(consume_orders)

        cache_shape = (*selected_steps.shape, *latent.shape[1:])
        cached_latents = torch.empty(cache_shape, device=latent.device, dtype=latent.dtype)
        cached_targets = torch.empty(cache_shape, device=latent.device, dtype=torch.float32)
        cached_sigmas = torch.empty(selected_steps.shape, device=latent.device, dtype=torch.float32)
        self.teacher.set_training(False)
        if student_rollout:
            self.student.set_training(False)
        try:
            for step_index, timestep in enumerate(scheduler.timesteps):
                sigma = wan_unipc_timestep_to_sigma(timestep, self.noise_scheduler.num_train_timesteps).reshape(1).expand(batch_size)
                selected_mask = selected_steps == step_index
                is_selected = bool(selected_mask.any())
                teacher_target = None
                if not student_rollout or is_selected:
                    teacher_target = self.teacher.predict_guided_velocity(latent, sigma, condition, negative_condition, self.guidance_scale, self.cfg_norm)
                if is_selected:
                    indices = selected_mask.nonzero(as_tuple=True)
                    batch_indices = indices[0]
                    cached_latents[indices] = latent[batch_indices]
                    cached_targets[indices] = teacher_target[batch_indices].float()
                    cached_sigmas[indices] = sigma[batch_indices]
                rollout_velocity = self.student.predict_velocity(latent, sigma, condition) if student_rollout else teacher_target
                latent = scheduler.step(rollout_velocity.float(), timestep, latent, return_dict=False)[0]
        finally:
            if student_rollout:
                self._restore_trainable_model(self.model)

        self._trajectory_cache = {
            "latents": cached_latents.detach(),
            "targets": cached_targets.detach(),
            "sigmas": cached_sigmas.detach(),
            "condition": condition,
            "negative_condition": negative_condition,
            "orders": consume_orders,
        }
        self._trajectory_cursor = 0
        logger.info("[train] refreshed {} UniPC trajectory cache batch={} steps={} intervals={}", self.rollout_policy, batch_size, self.trajectory_rollout_steps, self.trajectory_intervals)

    def _next_trajectory_batch(self, sample):
        if self._trajectory_cache is None or self._trajectory_cursor >= self.trajectory_refresh_iters:
            self._refresh_trajectory_cache(sample)
        cache = self._trajectory_cache
        batch_indices = torch.arange(cache["latents"].shape[0], device=cache["latents"].device)
        interval_indices = cache["orders"][batch_indices, self._trajectory_cursor]
        result = {
            "latents": cache["latents"][batch_indices, interval_indices],
            "targets": cache["targets"][batch_indices, interval_indices],
            "sigmas": cache["sigmas"][batch_indices, interval_indices],
            "condition": cache["condition"],
            "negative_condition": cache["negative_condition"],
            "interval_indices": interval_indices,
        }
        self._trajectory_cursor += 1
        return result

    @staticmethod
    def _prediction_mse(prediction, target):
        return (prediction.float() - target.float()).square().mean()

    def compute_loss_on_sample(self, sample):
        trajectory = self._next_trajectory_batch(sample)
        latent, sigma, target = trajectory["latents"], trajectory["sigmas"], trajectory["targets"]
        student_cond = self.student.predict_velocity(latent, sigma, trajectory["condition"])
        cond_loss = self._prediction_mse(student_cond, target)
        loss = cond_loss
        metrics = {
            "cond_loss": cond_loss.detach(),
            "sigma": sigma.float().mean(),
            "trajectory_interval": trajectory["interval_indices"].float().mean(),
        }
        use_uncond = self.uncond_enabled and self.uncond_loss_weight > 0
        if use_uncond and cond_loss.requires_grad:
            # Accumulation is restricted to one. Free this graph before the
            # optional second forward; the outer loop backpropagates that branch.
            cond_loss.backward()
            loss = cond_loss.detach()
            del student_cond
        if use_uncond:
            student_uncond = self.student.predict_velocity(latent, sigma, trajectory["negative_condition"])
            uncond_loss = self._prediction_mse(student_uncond, target)
            loss = loss + self.uncond_loss_weight * uncond_loss
            metrics["uncond_loss"] = uncond_loss.detach()
        return LossResult(loss=loss, metrics=metrics)
