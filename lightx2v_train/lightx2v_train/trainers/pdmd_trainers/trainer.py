import math
import shutil
from contextlib import nullcontext
from pathlib import Path

import torch
from loguru import logger

from lightx2v_train.runtime.checkpoint import prune_checkpoints
from lightx2v_train.runtime.distributed import barrier, get_rank, get_world_size, is_main_process, is_sequence_parallel_enabled, reduce_mean
from lightx2v_train.utils.registry import TRAINER_REGISTER

from ..dmd.trainer import DmdTrainer
from .core import (
    critic_objective,
    critic_sigmas,
    fake_updates_before,
    full_rollout,
    is_student_update,
    move_tree,
    rollout_grid,
    student_objective,
    student_prediction,
)
from .utils import fork_rng, map_latents, noise_like, seed_generator, seed_rng, stream_seed


@TRAINER_REGISTER("pdmd")
class PDMDTrainer(DmdTrainer):
    trainer_name = "pdmd"
    supports_ida = False
    supports_diversity_loss = False
    supports_real_data_fake = False

    def __init__(self, config):
        super().__init__(config)
        if self.random_schedule_enabled or self.warp_denoising_step:
            raise ValueError("PDMD uses an explicit physical rollout grid.")
        if is_sequence_parallel_enabled():
            raise ValueError("PDMD requires independent trajectories per rank; disable sequence parallelism.")
        self.seed = int(config.get("seed", 42))
        self.fake_steps = self.fake_update_ratio
        self.accum = int(self.gradient_accumulation_iters)
        stages = self.dmd_config.get("video_shift_schedule", [{"start": 0, "shift": self.dmd_config.get("rollout_video_shift", 1.0)}])
        self.video_shifts = [(int(stage["start"]), float(stage["shift"])) for stage in stages]
        self.audio_shift = float(self.dmd_config.get("rollout_audio_shift", 1.0))
        starts = [start for start, _ in self.video_shifts]
        shifts = [shift for _, shift in self.video_shifts] + [self.audio_shift]
        if not starts or starts[0] != 0 or starts != sorted(set(starts)) or any(start % (self.fake_steps + 1) for start in starts):
            raise ValueError("PDMD shift stages must start at zero and switch between fake/student cycles.")
        if any(not math.isfinite(shift) or shift <= 0 for shift in shifts):
            raise ValueError("PDMD rollout shifts must be positive and finite.")
        self.critic_video_threshold = float(self.dmd_config.get("critic_video_threshold", 0.95))
        self.critic_audio_floor = float(self.dmd_config.get("critic_audio_floor", 0.85))
        if not 0 <= self.critic_video_threshold <= 1 or not 0 <= self.critic_audio_floor <= 1:
            raise ValueError("PDMD critic threshold and floor must be in [0,1].")
        self.saved_trajectories = []

    def _optimizer_steps(self):
        fake = fake_updates_before(self.max_train_iters, self.fake_steps)
        return max(1, self.max_train_iters - fake), max(1, fake)

    def _recipe(self):
        sampler = self.dataloader_train.sampler
        return {
            "seed": self.seed,
            "model": self.model_config["name"],
            "world_size": get_world_size(),
            "manifest": self.dataloader_train.dataset.fingerprint,
            "sampler": [sampler.seed, sampler.shuffle, sampler.drop_last],
            "fake_steps": self.fake_steps,
            "accum": self.accum,
            "rollout_steps": self.num_inference_steps,
            "video_shifts": self.video_shifts,
            "audio_shift": self.audio_shift,
            "critic_video_threshold": self.critic_video_threshold,
            "critic_audio_floor": self.critic_audio_floor,
        }

    def _restore_trajectories(self, checkpoint, iteration):
        if checkpoint is None:
            return
        path = Path(checkpoint) / f"pdmd-rank-{get_rank():05d}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["iteration"] != iteration or state["recipe"] != self._recipe():
            raise ValueError(f"PDMD checkpoint does not match this run: {path}")
        self.saved_trajectories = move_tree(state["trajectories"], self.student.device)
        expected = self.accum if is_student_update(iteration, self.fake_steps) else 0
        if len(self.saved_trajectories) != expected:
            raise ValueError(f"PDMD checkpoint contains the wrong number of trajectories: {path}")

    def save_checkpoint(self, iteration, save_total_limit):
        root = Path(self.output_train_dir)
        output = root / f"checkpoint-{iteration:09d}"
        pending = root / f".{output.name}.pending"
        if is_main_process():
            if output.exists():
                raise FileExistsError(output)
            if pending.exists():
                shutil.rmtree(pending)
            pending.mkdir(parents=True)
        barrier()
        self.checkpoint_manager.save_checkpoint(iteration, None, save_dir=pending)
        state = {
            "iteration": iteration,
            "recipe": self._recipe(),
            "trajectories": move_tree(self.saved_trajectories, "cpu"),
        }
        torch.save(state, pending / f"pdmd-rank-{get_rank():05d}.pt")
        barrier()
        if is_main_process():
            pending.rename(output)
            if save_total_limit is not None:
                prune_checkpoints(root, save_total_limit + 1)
            logger.info("[pdmd] checkpoint ready: {}", output)
        barrier()

    def _trajectory(self, sample, iteration, micro, grid):
        with fork_rng(self.student.device) as devices:
            seed_rng(stream_seed(self.seed, "condition", iteration, get_rank(), micro), devices)
            conditions = self._encode_conditions(sample)
            seed_rng(stream_seed(self.seed, "noise", iteration, get_rank(), micro), devices)
            noise = self.sample_initial_latents(self._latent_shape(sample))
        return full_rollout(self.student, noise, conditions, grid)

    def _student_backward(self, trajectory, index, ratio, iteration, micro):
        recompute = self.gan is not None and self.gan.config.get("student_recompute", False)
        context = self.gan.frozen_critic() if self.gan is not None else nullcontext()
        if recompute:
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(self.student.device) if torch.device(self.student.device).type == "cuda" else None
        with context:
            loss, clean, metrics = student_objective(self, trajectory, index, ratio, recompute=recompute)
            if self.gan is not None:
                gan_loss, gan_metrics = self.gan.generator(clean, trajectory.conditions[0], iteration, micro)
                loss = loss + gan_loss
                metrics.update(gan_metrics)
            (loss / self.accum).backward()
            loss_value = loss.detach()
            if recompute:
                gradients = map_latents(clean, lambda value: value.grad.detach())
                del loss, clean, gan_loss
                # Replay after releasing the critic graph, with the original RNG state.
                with fork_rng(self.student.device):
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state(cuda_rng, self.student.device)
                    _, clean = student_prediction(self.student, trajectory, index)
                    if torch.is_tensor(clean):
                        torch.autograd.backward(clean, gradients)
                    else:
                        torch.autograd.backward((clean.video, clean.audio), (gradients.video, gradients.audio))
        return loss_value, metrics

    def _fake_backward(self, trajectory, sample, cursor, iteration, micro, generator):
        clean = trajectory.states[-1]
        sigmas = critic_sigmas(
            generator,
            video_threshold=self.critic_video_threshold,
            audio_floor=self.critic_audio_floor,
            device=self.student.device,
            joint=not torch.is_tensor(clean),
        )
        noise = noise_like(clean, seed_generator(self.seed, "renoise", iteration, get_rank(), micro, device=self.student.device))
        loss = critic_objective(self, trajectory, sigmas, noise)
        (loss / self.accum).backward()
        metrics = {"score_sigma_video": sigmas[0], "fake_flow_loss": loss.detach()}
        if len(sigmas) == 2:
            metrics["score_sigma_audio"] = sigmas[1]
        value = loss.detach()
        del loss, noise
        if self.gan is not None:
            gan_loss, gan_metrics = self.gan.backward_discriminator(clean, trajectory.conditions[0], sample, cursor, iteration, micro, 1 / self.accum)
            value = value + gan_loss
            metrics.update(gan_metrics)
        return value, metrics

    def _update(self, samples, iteration, grid):
        student_update = is_student_update(iteration, self.fake_steps)
        if student_update:
            optimizer, scheduler = self.optimizer, self.lr_scheduler
            params, set_sync = self.trainable_params, self._set_student_gradient_sync
        else:
            optimizer, scheduler = self.fake_optimizer, self.fake_lr_scheduler
            params, set_sync = self.fake_trainable_params, self._set_fake_gradient_sync
        optimizer.zero_grad(set_to_none=True)

        cursor = fake_updates_before(iteration, self.fake_steps) * self.accum
        index = int(torch.randint(self.num_inference_steps, (), generator=seed_generator(self.seed, "exit", iteration)))
        totals = {}
        for micro in range(self.accum):
            set_sync(micro == self.accum - 1)
            draws = seed_generator(self.seed, "draws", iteration, get_rank(), micro)
            if student_update:
                trajectory = self.saved_trajectories[micro]
                ratio = float(torch.rand((), generator=draws))
                loss, metrics = self._student_backward(trajectory, index, ratio, iteration, micro)
            else:
                sample = next(samples)
                trajectory = self._trajectory(sample, iteration, micro, grid)
                if is_student_update(iteration + 1, self.fake_steps):
                    self.saved_trajectories.append(trajectory)
                loss, metrics = self._fake_backward(trajectory, sample, cursor + micro, iteration, micro, draws)
            for key, value in dict(loss=loss, **metrics).items():
                totals[key] = totals.get(key, 0.0) + float(value) / self.accum

        gan_update = self.gan is not None and not student_update
        norm = self.gan.clip_discriminator_gradients(self.max_grad_norm) if gan_update else torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        norm = self._check_finite_gradients(norm, totals["loss"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if gan_update:
            self.gan.step()
        if student_update:
            self.saved_trajectories = []
        totals["grad_norm"] = float(norm)
        return "student" if student_update else "fake", totals

    def train(self):
        checkpoint, iteration = self._resolve_resume()
        if iteration >= self.max_train_iters:
            return
        self._restore_trajectories(checkpoint, iteration)
        cursor = fake_updates_before(iteration, self.fake_steps) * self.accum
        self.dataloader_train.sampler.set_start(cursor)
        self.setup(resume_ckpt_path=checkpoint)
        samples = self._iter_train_samples()
        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_val)
        if is_main_process():
            Path(self.output_train_dir).mkdir(parents=True, exist_ok=True)
        barrier()

        grid = None
        video_shift = None
        for update in range(iteration, self.max_train_iters):
            shift = next(shift for start, shift in reversed(self.video_shifts) if update >= start)
            if shift != video_shift:
                video_shift = shift
                grid = rollout_grid(self.num_inference_steps, shift, self.audio_shift, device=self.student.device)
                logger.info("[pdmd] update={} video_shift={} audio_shift={} sigmas={}", update, shift, self.audio_shift, grid.tolist())
            role, totals = self._update(samples, update, grid)
            completed = update + 1
            if completed == 1 or completed % self.train_log_every_iters == 0 or completed == self.max_train_iters:
                metrics = {f"train/{key}": reduce_mean(value) for key, value in totals.items()}
                logger.info("[pdmd] update={}/{} role={} student_updates={} metrics={}", completed, self.max_train_iters, role, completed // (self.fake_steps + 1), metrics)
                self.log_metrics(metrics, step=completed)
            if completed == self.max_train_iters or (self.save_every_iters and completed % self.save_every_iters == 0):
                self.save_checkpoint(completed, self.save_total_limit)
            if self.infer_every_iters and completed % self.infer_every_iters == 0:
                self.run_inference(completed)
        self.finish_monitor()
