"""Opt-in official H3 PDMD optimizer-update loop using existing model roles.

The user's Ref cache, FSDP2 topology, full critic, LoRA scaling and role dtypes
are intentional adaptations. Default DMD/head/DMAD training never enters here.
"""

import math
import os
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_train.runtime.distributed import barrier, get_world_size, is_main_process, reduce_mean

from .official_pdmd_core import (
    critic_objective,
    critic_sigmas,
    critic_updates_before,
    full_rollout,
    joint_noise_like,
    role_at,
    rollout_grid,
    stream_seed,
    student_objective,
)


def move_tensors(value, device):
    """Preserve packed-reference dataclasses and tensor dtypes in rank state."""
    if torch.is_tensor(value):
        return value.detach().to(device=device)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: move_tensors(getattr(value, field.name), device) for field in fields(value)})
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    return value


def validate_official_pdmd_config(config):
    """Validate the opt-in flow without constraining the user's role dtypes."""
    training = config["training"]
    dmd = training.get("dmd", {})
    matching = config["model"].get("capabilities", {}).get("distribution_matching", {})
    if training.get("method") != "dmd" or config["model"]["name"] != "minimax_h3_ref2av":
        raise ValueError("official_pdmd currently supports the H3 Ref2AV DMD trainer with a Ref cost sampler only")
    if not matching.get("official_pdmd", False):
        raise ValueError("training.dmd.official_pdmd requires model.capabilities.distribution_matching.official_pdmd=true")
    if dmd.get("update_order", "fake_first") != "fake_first":
        raise ValueError("official_pdmd requires fake_first update order")
    if dmd.get("model_mode", "eval") != "eval":
        raise ValueError("official_pdmd requires eval model mode (autograd remains enabled)")
    for feature in ("residual_head", "div_loss", "fake_real", "ida", "random_schedule"):
        if dmd.get(feature, {}).get("enabled", False):
            raise ValueError(f"official_pdmd cannot be combined with {feature}")
    if training.get("student", {}).get("ema", {}).get("enabled", False):
        raise ValueError("official_pdmd uses live student weights, not EMA")
    if float(training.get("teacher", {}).get("guidance_scale", 1.0)) != 1.0:
        raise ValueError("official H3 PDMD has no CFG branch; guidance_scale must be 1")
    if (matching.get("adaptive_video_regularization") or {}).get("enabled", False):
        raise ValueError("official_pdmd does not add adaptive video regularization")
    sp = config.get("distributed", {}).get("sequence_parallel", {})
    if sp.get("enabled", False) and int(sp.get("size", 1)) != 1:
        raise ValueError("official_pdmd currently requires sequence_parallel size 1")
    if int(training.get("gradient_accumulation_iters", 1)) < 1:
        raise ValueError("gradient_accumulation_iters must be positive")


class OfficialH3PdmdTraining:
    state_version = 1

    def __init__(self, trainer):
        self.trainer = trainer
        self.dmd = trainer.dmd_config
        self.critic_steps = int(trainer.fake_update_ratio)
        self.accum = int(trainer.gradient_accumulation_iters)
        self.seed = int(trainer.config.get("seed", 42))
        self.cached = []
        self.consumed_microbatches = 0
        validate_official_pdmd_config(trainer.config)

    def recipe(self):
        config = self.trainer.config
        training = {key: value for key, value in config["training"].items() if key not in {"max_train_iters", "output_dir", "save_every_iters", "save_total_limit", "save_consolidated_weights"}}
        data = {key: value for key, value in config["data"]["train"].items() if key not in {"num_workers", "pin_memory"}}
        return {"seed": self.seed, "model": config["model"], "training": training, "data": data, "scheduler": config["scheduler"], "distributed": config.get("distributed", {})}

    @staticmethod
    def rank():
        return dist.get_rank() if dist.is_initialized() else 0

    def _load_cursor(self, checkpoint, iteration):
        if checkpoint is None:
            return
        marker = Path(checkpoint) / "official_pdmd.complete"
        if not marker.is_file():
            raise RuntimeError(f"Not a complete official-PDMD checkpoint: {checkpoint}; use a fresh output for this recipe")
        path = Path(checkpoint) / f"official_pdmd.rank{self.rank():05d}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        expected = {"version": self.state_version, "iteration": iteration, "rank": self.rank(), "world_size": get_world_size(), "recipe": self.recipe()}
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(f"Official PDMD checkpoint {key} mismatch in {path}; use a fresh output directory")
        consumed = critic_updates_before(iteration, self.critic_steps) * self.accum
        if state.get("consumed_microbatches") != consumed:
            raise RuntimeError(f"Official PDMD data cursor mismatch in {path}")
        self.consumed_microbatches = consumed
        self.cached = state["cached"]
        needed = self.accum if role_at(iteration, self.critic_steps) == "student" else 0
        if len(self.cached) != needed:
            raise RuntimeError(f"Official PDMD preceding-critic trajectory missing or stale in {path}")

    def _save(self, iteration):
        t = self.trainer
        directory = Path(t.output_train_dir) / f"checkpoint-{iteration:09d}"
        state = {
            "version": self.state_version,
            "iteration": iteration,
            "rank": self.rank(),
            "world_size": get_world_size(),
            "recipe": self.recipe(),
            "consumed_microbatches": self.consumed_microbatches,
            "cached": move_tensors(self.cached, "cpu"),
        }
        # Complete normal model/optimizer state first. An interrupted save with
        # no complete marker is rejected, never silently resumed as legacy DMD.
        t.save_checkpoint(iteration, t.save_total_limit)
        pending = directory / f".official_pdmd.rank{self.rank():05d}.tmp"
        torch.save(state, pending)
        os.replace(pending, directory / f"official_pdmd.rank{self.rank():05d}.pt")
        barrier()
        if is_main_process():
            (directory / "official_pdmd.complete").write_text(f"{iteration}\n", encoding="utf-8")
        barrier()

    def _trajectory(self, sample, iteration, slot):
        t = self.trainer
        device = torch.device(t.student.device)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        # Ref conditioning has its own noise augmentation. Isolate it from
        # data-loader prefetch and from each optimizer role's noise streams.
        with torch.random.fork_rng(devices=cuda_devices):
            seed = stream_seed(self.seed, "condition", iteration, slot)
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                torch.cuda.default_generators[cuda_devices[0]].manual_seed(seed)
            condition, _ = t._encode_conditions(sample)
            shape = t._latent_shape(sample)
            seed = stream_seed(self.seed, "noise", iteration, slot)
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                torch.cuda.default_generators[cuda_devices[0]].manual_seed(seed)
            noise = t.sample_initial_latents(shape)
        return full_rollout(t.student, noise, condition, self.video_grid, self.audio_grid)

    def _update(self, samples, iteration):
        t = self.trainer
        role = role_at(iteration, self.critic_steps)
        if role == "student":
            optimizer, scheduler, params, set_sync = t.optimizer, t.lr_scheduler, t.trainable_params, t._set_student_gradient_sync
            if len(self.cached) != self.accum:
                raise RuntimeError("Student update requires the immediately preceding critic trajectories")
        else:
            optimizer, scheduler, params, set_sync = t.fake_optimizer, t.fake_lr_scheduler, t.fake_trainable_params, t._set_fake_gradient_sync
            self.cached = []
        t.optimizer.zero_grad(set_to_none=True)
        t.fake_optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(stream_seed(self.seed, "exit", iteration))
        index = int(torch.randint(t.num_inference_steps, (1,), generator=generator))
        totals = {}
        for micro in range(self.accum):
            slot = micro * get_world_size() + self.rank()
            if role == "student":
                trajectory = self.cached[micro]
            else:
                trajectory = self._trajectory(next(samples), iteration, slot)
                self.consumed_microbatches += 1
            set_sync(micro == self.accum - 1)
            draws = torch.Generator().manual_seed(stream_seed(self.seed, "draws", iteration, slot))
            if role == "student":
                ratio = float(torch.rand((), generator=draws))
                loss, metrics = student_objective(t.student, t.fake, t.teacher, trajectory, index, ratio)
            else:
                sigmas = critic_sigmas(draws, video_threshold=float(self.dmd.get("critic_video_threshold", 0.95)), audio_floor=float(self.dmd.get("critic_audio_floor", 0.85)), device=t.student.device)
                noise_generator = torch.Generator(device=t.student.device).manual_seed(stream_seed(self.seed, "renoise", iteration, slot))
                fresh = joint_noise_like(trajectory.states[-1], noise_generator)
                loss = critic_objective(t.student, t.fake, trajectory, sigmas, fresh)
                metrics = {"score_sigma_video": sigmas[0], "score_sigma_audio": sigmas[1]}
                if role_at(iteration + 1, self.critic_steps) == "student":
                    self.cached.append(trajectory)
            (loss / self.accum).backward()
            for name, value in {"loss": loss.detach(), **metrics}.items():
                value = value.detach().item() if torch.is_tensor(value) else float(value)
                totals[name] = totals.get(name, 0.0) + value / self.accum
            del loss, trajectory
        t._sync_sequence_parallel_grads(params)
        norm = torch.nn.utils.clip_grad_norm_(params, t.max_grad_norm)
        if hasattr(norm, "full_tensor"):
            norm = norm.full_tensor()
        finite = torch.isfinite(norm).to(device=t.student.device, dtype=torch.int32)
        if not all(math.isfinite(value) for value in totals.values()):
            finite.zero_()
        if dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError(f"official_pdmd nonfinite {role} gradient norm or metrics at optimizer update {iteration + 1}")
        optimizer.step()
        if role == "student":
            t._after_student_optimizer_step("main")
            self.cached = []
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        totals["grad_norm"] = float(norm)
        return role, totals

    def run(self):
        t = self.trainer
        checkpoint, iteration = t._resolve_resume()
        self._load_cursor(checkpoint, iteration)
        sampler = t.dataloader_train.sampler
        if not hasattr(sampler, "configure_from_microbatch_offset"):
            raise ValueError("official H3 PDMD currently requires the Ref cost sampler with an explicit microbatch cursor")
        sampler.configure_from_microbatch_offset(start_microbatch=self.consumed_microbatches, gradient_accumulation_iters=self.accum, fake_update_ratio=self.critic_steps)
        t.setup(resume_ckpt_path=checkpoint)
        self.cached = move_tensors(self.cached, t.student.device)
        self.video_grid = rollout_grid(t.num_inference_steps, float(self.dmd.get("official_rollout_video_shift", 1.0)), t.student.device)
        self.audio_grid = rollout_grid(t.num_inference_steps, float(self.dmd.get("official_rollout_audio_shift", 1.0)), t.student.device)
        if is_main_process():
            os.makedirs(t.output_train_dir, exist_ok=True)
        barrier()
        logger.info(
            "[pdmd_official] optimizer_updates={}/{} critic_steps={} global_batch={} rollout_video={} rollout_audio={} data_microbatch_offset={}",
            iteration,
            t.max_train_iters,
            self.critic_steps,
            self.accum * get_world_size(),
            self.video_grid.tolist(),
            self.audio_grid.tolist(),
            self.consumed_microbatches,
        )
        samples = t._iter_train_samples()
        last_saved = iteration if checkpoint is not None else -1
        while iteration < t.max_train_iters:
            t.student.on_iteration_start(iteration)
            role, metrics = self._update(samples, iteration)
            iteration += 1
            t.student.on_iteration_end(iteration)
            if iteration == 1 or iteration % t.train_log_every_iters == 0 or iteration == t.max_train_iters:
                metrics = {name: reduce_mean(value) for name, value in metrics.items()}
                logger.info(
                    "[pdmd_official] update={}/{} role={} loss={:.6f} student_updates={} critic_updates={} consumed_microbatches={} metrics={}",
                    iteration,
                    t.max_train_iters,
                    role,
                    metrics["loss"],
                    iteration // (self.critic_steps + 1),
                    critic_updates_before(iteration, self.critic_steps),
                    self.consumed_microbatches,
                    metrics,
                )
                t.log_metrics({f"train/{role}_{key}": value for key, value in metrics.items()}, step=iteration)
            if t.save_every_iters and iteration % t.save_every_iters == 0:
                self._save(iteration)
                last_saved = iteration
        if iteration != last_saved:
            self._save(iteration)
        logger.info("[pdmd_official] finished optimizer_updates={}/{}", iteration, t.max_train_iters)
