"""Official PDMD optimizer-update loop using H3 video/image or Wan model roles.

The user's Ref cache, FSDP2 topology, full critic, LoRA scaling and role dtypes
are intentional adaptations. Default DMD/head/DMAD training never enters here.
"""

import hashlib
import json
import math
import os
from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_train.runtime.distributed import barrier, get_data_parallel_rank, get_data_parallel_world_size, get_world_size, is_main_process, reduce_mean

from .official_pdmd_core import (
    PdmdCondition,
    critic_objective,
    critic_sigmas,
    critic_updates_before,
    full_rollout,
    joint_map,
    joint_noise_like,
    role_at,
    rollout_grid,
    stream_seed,
    student_objective,
    student_prediction,
)
from .rollout_schedule import rollout_shifts, validate_rollout_schedule


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
    model_name = config["model"]["name"]
    if training.get("method") != "dmd" or model_name not in {"minimax_h3_ref2av", "minimax_h3_t2av", "minimax_h3_image", "wan_t2v"}:
        raise ValueError("official_pdmd requires an H3 video/image or Wan T2V DMD model")
    if config["model"]["name"] == "minimax_h3_t2av":
        data = config["data"]["train"]
        legacy = data.get("name") == "minimax_h3_cache_dataset" and data.get("route_mode") == "stratified"
        curriculum = data.get("name") == "minimax_h3_curriculum_cache_dataset" and data.get("kind") == "fl2av"
        if not (legacy or curriculum or data.get("name") == "h3_selfgen_manifest"):
            raise ValueError("official_pdmd base/keyframe tasks require a stratified H3 cache or an FL2AV new-then-old curriculum cache")
    elif config["data"]["train"].get("name") == "minimax_h3_curriculum_cache_dataset" and config["data"]["train"].get("kind") != "ref2av":
        raise ValueError("official_pdmd Ref2AV requires kind=ref2av for a curriculum cache")
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
    if model_name != "wan_t2v" and float(training.get("teacher", {}).get("guidance_scale", 1.0)) != 1.0:
        raise ValueError("official H3 PDMD has no CFG branch; guidance_scale must be 1")
    if model_name in {"wan_t2v", "minimax_h3_image"} and int(config["data"]["train"].get("batch_size", 1)) != 1:
        raise ValueError("official single-modality PDMD currently requires singleton per-rank batches")
    explicit_sigmas = dmd.get("official_rollout_sigmas")
    if model_name == "minimax_h3_image" and explicit_sigmas is None:
        raise ValueError("H3-Image PDMD requires explicit physical official_rollout_sigmas")
    if explicit_sigmas is not None:
        if model_name not in {"wan_t2v", "minimax_h3_image"}:
            raise ValueError("official_rollout_sigmas is supported only for single-modality PDMD")
        rollout_grid(int(dmd["num_inference_steps"]), sigmas=explicit_sigmas)
    if (matching.get("adaptive_video_regularization") or {}).get("enabled", False):
        raise ValueError("official_pdmd does not add adaptive video regularization")
    sp = config.get("distributed", {}).get("sequence_parallel", {})
    if sp.get("enabled", False) and int(sp.get("size", 1)) != 1:
        raise ValueError("official_pdmd currently requires sequence_parallel size 1")
    if int(training.get("gradient_accumulation_iters", 1)) < 1:
        raise ValueError("gradient_accumulation_iters must be positive")
    from .dmd2_gan import validate_gan_config

    validate_gan_config(config)
    validate_rollout_schedule(config)


class OfficialH3PdmdTraining:
    state_version = 1

    def __init__(self, trainer):
        self.trainer = trainer
        self.dmd = trainer.dmd_config
        self.critic_steps = int(trainer.fake_update_ratio)
        self.accum = int(trainer.gradient_accumulation_iters)
        self.seed = int(trainer.config.get("seed", 42))
        self.cached = []
        self.active_rollout_shifts = None
        self.consumed_microbatches = 0
        self.route_loss_scale = lambda microbatch: 1.0
        self.gan = None
        self.gan_real_stream = None
        validate_official_pdmd_config(trainer.config)
        self.paired_manifest = trainer.config["data"]["train"].get("name") == "h3_selfgen_manifest"
        if self.dmd.get("gan", {}).get("enabled", False) and not self.paired_manifest:
            from .dmd2_gan_data import DMD2GanRealStream

            model_name = trainer.config["model"]["name"]
            task = {"minimax_h3_image": "t2i", "minimax_h3_ref2av": "ref2av"}.get(model_name, "t2av")
            geometry = None if task == "ref2av" else tuple(self.dmd["generation_shapes"][0]["value"])
            self.gan_real_stream = DMD2GanRealStream(
                self.dmd["gan"]["real_cache"],
                seed=self.seed,
                rank=get_data_parallel_rank(),
                world_size=get_data_parallel_world_size(),
                geometry=geometry,
                task=task,
            )

    def recipe(self):
        config = self.trainer.config
        training = {key: value for key, value in config["training"].items() if key not in {"max_train_iters", "output_dir", "save_every_iters", "save_total_limit", "save_consolidated_weights"}}
        data = {key: value for key, value in config["data"]["train"].items() if key not in {"num_workers", "pin_memory"}}
        recipe = {"seed": self.seed, "model": config["model"], "training": training, "data": data, "scheduler": config["scheduler"], "distributed": config.get("distributed", {})}
        if config["model"]["name"] in {"wan_t2v", "minimax_h3_image"}:
            dataset = self.trainer.dataloader_train.dataset
            digest = hashlib.sha256()
            for record in dataset.samples:
                digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))
                digest.update(b"\n")
            recipe["dataset"] = {"rows": len(dataset), "sha256": digest.hexdigest()}
        if self.paired_manifest:
            recipe["manifest_sha256"] = self.trainer.dataloader_train.dataset.digest
        if self.gan_real_stream is not None:
            recipe["gan_real_data"] = self.gan_real_stream.recipe()
            samples = self.trainer.dataloader_train.dataset.samples
            digest = hashlib.sha256()
            for sample in samples:
                digest.update(json.dumps(sample, sort_keys=True, ensure_ascii=False, default=str).encode())
                digest.update(b"\n")
            recipe["gan_prompt_data"] = {"rows": len(samples), "sha256": digest.hexdigest()}
        return recipe

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
            if self.dmd.get("rollout_shift_schedule"):
                (directory / "rollout_schedule.json").write_text(json.dumps({
                    "completed_updates": iteration,
                    "num_inference_steps": t.num_inference_steps,
                    "last_update_shifts": self.active_rollout_shifts,
                    "next_update_shifts": rollout_shifts(self.dmd, iteration),
                    "schedule": self.dmd["rollout_shift_schedule"],
                }, indent=2) + "\n", encoding="utf-8")
            (directory / "official_pdmd.complete").write_text(f"{iteration}\n", encoding="utf-8")
        barrier()

    def _set_rollout_schedule(self, completed_updates):
        shifts = rollout_shifts(self.dmd, completed_updates)
        if shifts == self.active_rollout_shifts:
            return
        if self.active_rollout_shifts is not None and self.cached:
            raise RuntimeError("Cannot replace a rollout schedule while a student trajectory is pending")
        t = self.trainer
        self.video_grid = rollout_grid(t.num_inference_steps, shifts[0], t.student.device,
                                       sigmas=self.dmd.get("official_rollout_sigmas"))
        self.audio_grid = rollout_grid(t.num_inference_steps, shifts[1], t.student.device)
        if self.dmd.get("rollout_shift_schedule"):
            # Export/validate the shifts used by this checkpoint without changing
            # the immutable config that identifies the full training recipe.
            t.student.video_shift, t.student.audio_shift = shifts
            logger.info("[pdmd_official][schedule] next_update={} video_shift={} audio_shift={}", completed_updates + 1, *shifts)
        self.active_rollout_shifts = shifts

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
            condition, negative = t._encode_conditions(sample)
            if negative is not None:
                condition = PdmdCondition(condition, negative, float(t.guidance_scale), t.cfg_norm)
            shape = t._latent_shape(sample)
            seed = stream_seed(self.seed, "noise", iteration, slot)
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                torch.cuda.default_generators[cuda_devices[0]].manual_seed(seed)
            noise = t.sample_initial_latents(shape)
        return full_rollout(t.student, noise, condition, self.video_grid, self.audio_grid)

    def _student_recompute_backward(self, trajectory, index, ratio, iteration, slot, route_scale):
        """Backpropagate through the critic and student sequentially.

        Scores and the GAN loss are evaluated once. Only the student forward
        is replayed, using the same RNG state, after the critic graph is freed.
        The caller keeps the critic frozen throughout both backwards.
        """
        t = self.trainer
        device = torch.device(t.student.device)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(cuda_devices[0]) if cuda_devices else None
        loss, metrics, clean = student_objective(t.student, t.fake, t.teacher, trajectory, index, ratio, return_clean=True, detach_student=True)
        metrics["pdmd_loss"] = loss.detach()
        gan_loss, gan_metrics = self.gan.generator(clean, trajectory.condition, iteration, slot)
        metrics.update(gan_metrics)
        loss = (loss + gan_loss) * route_scale
        (loss / self.accum).backward()
        gradients = joint_map(clean, lambda tensor, _: tensor.grad.detach())
        loss_value = loss.detach()
        del clean, loss, gan_loss

        # Replay consumes the first forward's RNG state without advancing the
        # process RNG a second time or repeating teacher/fake score queries.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.set_rng_state(cpu_rng)
            if cuda_devices:
                torch.cuda.set_rng_state(cuda_rng, cuda_devices[0])
            _, clean, _ = student_prediction(t.student, trajectory, index)
            outputs = (clean,) if torch.is_tensor(clean) else (clean.video, clean.audio)
            grad_tensors = (gradients,) if torch.is_tensor(gradients) else (gradients.video, gradients.audio)
            torch.autograd.backward(outputs, grad_tensors)
        return loss_value, metrics

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
        if self.gan is not None:
            t.dmd2_gan_optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(stream_seed(self.seed, "exit", iteration))
        index = int(torch.randint(t.num_inference_steps, (1,), generator=generator))
        totals = {}
        batch_start = self.consumed_microbatches - self.accum if role == "student" else self.consumed_microbatches
        for micro in range(self.accum):
            slot = micro * get_world_size() + self.rank()
            if role == "student":
                trajectory = self.cached[micro]
            else:
                sample = next(samples)
                trajectory = self._trajectory(sample, iteration, slot)
                self.consumed_microbatches += 1
            set_sync(micro == self.accum - 1)
            draws = torch.Generator().manual_seed(stream_seed(self.seed, "draws", iteration, slot))
            route_scale = self.route_loss_scale(batch_start + micro)
            if role == "student":
                ratio = float(torch.rand((), generator=draws))
                if self.gan is not None:
                    from .dmd2_gan_head import frozen_dmd2_gan_critic

                    context = frozen_dmd2_gan_critic(t)
                else:
                    context = nullcontext()
                # Freeze parameters through backward too: activation checkpoint
                # recomputation must see the same flags as the forward pass.
                with context:
                    if self.gan is not None and self.dmd["gan"].get("student_recompute", False):
                        loss, metrics = self._student_recompute_backward(trajectory, index, ratio, iteration, slot, route_scale)
                    else:
                        if self.gan is None:
                            loss, metrics = student_objective(t.student, t.fake, t.teacher, trajectory, index, ratio)
                        else:
                            loss, metrics, clean = student_objective(t.student, t.fake, t.teacher, trajectory, index, ratio, return_clean=True)
                            metrics["pdmd_loss"] = loss.detach()
                            gan_loss, gan_metrics = self.gan.generator(clean, trajectory.condition, iteration, slot)
                            loss = loss + gan_loss
                            metrics.update(gan_metrics)
                            del clean, gan_loss
                        loss = loss * route_scale
                        (loss / self.accum).backward()
                loss_value = loss.detach()
            else:
                video_only = torch.is_tensor(trajectory.states[-1])
                sigmas = critic_sigmas(
                    draws,
                    video_threshold=float(self.dmd.get("critic_video_threshold", 0.95)),
                    audio_floor=float(self.dmd.get("critic_audio_floor", 0.85)),
                    device=t.student.device,
                    video_only=video_only,
                )
                noise_generator = torch.Generator(device=t.student.device).manual_seed(stream_seed(self.seed, "renoise", iteration, slot))
                fresh = joint_noise_like(trajectory.states[-1], noise_generator)
                loss = critic_objective(t.student, t.fake, trajectory, sigmas, fresh)
                metrics = {"score_sigma_video": sigmas[0]}
                if not video_only:
                    metrics["score_sigma_audio"] = sigmas[1]
                if role_at(iteration + 1, self.critic_steps) == "student":
                    self.cached.append(trajectory)
                loss = loss * route_scale
                (loss / self.accum).backward()
                loss_value = float(loss.detach())
                if self.gan is not None:
                    metrics["fake_flow_loss"] = loss_value
                    # Separate backwards avoid retaining three H3 graphs.
                    gan_value, gan_metrics = self.gan.backward_discriminator(
                        trajectory,
                        batch_start + micro,
                        iteration,
                        slot,
                        route_scale / self.accum,
                        real_sample=sample,
                    )
                    loss_value += route_scale * gan_value
                    metrics.update(gan_metrics)
            for name, value in {"loss": loss_value, **metrics}.items():
                value = value.detach().item() if torch.is_tensor(value) else float(value)
                totals[name] = totals.get(name, 0.0) + value / self.accum
            del loss, trajectory
        t._sync_sequence_parallel_grads(params)
        max_norm = float("inf") if t.max_grad_norm == 0 else t.max_grad_norm
        if role == "fake" and self.gan is not None:
            from .dmd2_gan_head import clip_dmd2_gan_grad_norm

            norm = clip_dmd2_gan_grad_norm(t, max_norm)
        else:
            norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
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
        if role == "fake" and self.gan is not None:
            t.dmd2_gan_optimizer.step()
            t.dmd2_gan_lr_scheduler.step()
            t.dmd2_gan_optimizer.zero_grad(set_to_none=True)
        if role == "student":
            t._after_student_optimizer_step("main")
            self.cached = []
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        totals["grad_norm"] = float(norm)
        return role, totals

    def _prepare_wan_data(self):
        """Install an absolute DP cursor for single-modality cached/online data."""
        t = self.trainer
        if t.config["model"]["name"] not in {"wan_t2v", "minimax_h3_image"} or hasattr(t.dataloader_train.sampler, "configure_from_microbatch_offset"):
            return
        from torch.utils.data import DataLoader

        from ..turbo.cursor import TurboCursorSampler

        loader = t.dataloader_train
        sampler = TurboCursorSampler(loader.dataset, seed=self.seed, shuffle=t.config["data"]["train"].get("shuffle", True), num_replicas=get_data_parallel_world_size(), rank=get_data_parallel_rank())
        t.dataloader_train = DataLoader(loader.dataset, batch_size=1, sampler=sampler, num_workers=loader.num_workers, pin_memory=loader.pin_memory, collate_fn=loader.collate_fn)

    def _validate_wan_residency(self):
        t = self.trainer
        if t.config["model"]["name"] != "wan_t2v" or t.model.running_dtype != torch.float32:
            return
        components = {name: role.denoiser() for name, role in (("student", t.student), ("fake", t.fake), ("teacher", t.teacher))}
        for name, wrapper in (("vae", t.model.vae), ("t5", t.model.text_encoder)):
            if wrapper is not None:
                components[name] = wrapper.model
        for name, module in components.items():
            parameters = list(module.parameters())
            dtypes = {parameter.dtype for parameter in parameters if parameter.is_floating_point()}
            devices = {parameter.device for parameter in parameters}
            if dtypes != {torch.float32} or devices != {t.model.device}:
                raise ValueError(f"Wan official PDMD requires resident FP32 {name}; got devices={devices}, dtypes={dtypes}")
            logger.info("[pdmd_official] resident component={} device={} dtype=fp32 parameters={}", name, t.model.device, sum(parameter.numel() for parameter in parameters))

    def _infer(self, iteration):
        t = self.trainer
        try:
            t.run_inference(iteration)
        finally:
            # Shared inference restores train mode; the released recipe uses eval.
            t.student.set_training(False)

    def run(self):
        t = self.trainer
        self._prepare_wan_data()
        checkpoint, iteration = t._resolve_resume()
        self._load_cursor(checkpoint, iteration)
        sampler = t.dataloader_train.sampler
        if not hasattr(sampler, "configure_from_microbatch_offset"):
            raise ValueError("official PDMD requires a sampler with an explicit microbatch cursor")
        sampler.configure_from_microbatch_offset(start_microbatch=self.consumed_microbatches, gradient_accumulation_iters=self.accum, fake_update_ratio=self.critic_steps)
        self.route_loss_scale = getattr(sampler, "cursor_microbatch_loss_scale", self.route_loss_scale)
        # A saved checkpoint describes the last completed update. Select that
        # phase before restoring metadata, then advance before the next update.
        self._set_rollout_schedule(max(0, iteration - 1))
        t.setup(resume_ckpt_path=checkpoint)
        gan_objective = None
        if self.dmd.get("gan", {}).get("enabled", False):
            from .dmd2_gan import Dmd2GanObjective, SingleHeadDmadObjective

            objective = self.dmd["gan"].get("objective", "dmd2")
            cls = SingleHeadDmadObjective if objective == "dmad_single_head" else Dmd2GanObjective
            gan_objective = cls(t, self.gan_real_stream)
            logger.info(
                "[pdmd_official][gan] objective={} shared fake backbone; G={} D={} student_recompute={} real_data={}",
                objective,
                gan_objective.generator_weight,
                gan_objective.discriminator_weight,
                self.dmd["gan"].get("student_recompute", False),
                "same manifest row as rollout" if self.paired_manifest else self.gan_real_stream.recipe(),
            )
        self._validate_wan_residency()
        self.cached = move_tensors(self.cached, t.student.device)
        if getattr(t, "infer_every_iters", None):
            t.inferencer.set_data(t.dataloader_val)
            if iteration == 0 and getattr(t, "infer_config", {}).get("run_at_start", False):
                self._infer(0)
            elif checkpoint is not None and iteration == t.max_train_iters and iteration % t.infer_every_iters == 0:
                # The final model/cursor save can complete before its preview
                # fails. Resume must retry that preview without another update.
                self._infer(iteration)
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
            self.gan = gan_objective if iteration >= self.dmd.get("gan", {}).get("start_after_updates", 0) else None
            self._set_rollout_schedule(iteration)
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
            if getattr(t, "infer_every_iters", None) and iteration % t.infer_every_iters == 0:
                self._infer(iteration)
        if iteration != last_saved:
            self._save(iteration)
        logger.info("[pdmd_official] finished optimizer_updates={}/{}", iteration, t.max_train_iters)
