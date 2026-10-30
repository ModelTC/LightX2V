"""H3 adaptation of TurboDiffusion's released rCM objectives and curriculum.

The LTX-2 training loop is not public. The dCM/sCM/DMD mathematics comes from
the released Wan rCM trainer; the dCM -> sCM -> joint curriculum comes from
TurboT2VA. Existing H3 role loading, FSDP, LoRA, EMA and DCP remain shared.
"""

import copy
import hashlib
import json
import math
import shutil
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_train.runtime.distributed import barrier, get_data_parallel_rank, get_data_parallel_world_size, get_world_size, is_main_process, reduce_mean
from lightx2v_train.utils.registry import TRAINER_REGISTER

from ..dmd.official_pdmd_core import stream_seed
from ..dmd.trainer import DmdTrainer
from . import core
from .config import TurboConfig
from .sampler import TurboCondition, component, joint_like, modality_names, noise_like, predict_trig, sample_turbo_latents


@TRAINER_REGISTER("turbo")
class TurboTrainer(DmdTrainer):
    """Count student updates explicitly; joint fake updates are additional."""

    trainer_name = "turbo"
    supports_ida = False
    supports_diversity_loss = False
    supports_real_data_fake = False
    checkpoint_version = 1

    def __init__(self, config):
        self.turbo = TurboConfig.from_mapping(config)
        self.public_config = copy.deepcopy(config)
        # Reuse the role/optimizer/checkpoint runtime. This internal mapping is
        # not a user-facing DMD recipe and cannot enter the official PDMD loop.
        internal = copy.deepcopy(config)
        internal["training"]["dmd"] = {
            "num_inference_steps": self.turbo.num_inference_steps,
            "fake_update_ratio": self.turbo.fake_update_ratio,
            "latent_dtype": "fp32",
            "model_mode": "eval",
        }
        if self.turbo.generation_shapes is not None:
            internal["training"]["dmd"]["generation_shapes"] = self.turbo.generation_shapes
        super().__init__(internal)
        self.consumed_microbatches = 0
        self.consumed_cm_microbatches = 0
        self.fake_updates = 0
        self.route_loss_scale = lambda _: 1.0
        self.cm_route_loss_scale = lambda _: 1.0

    @property
    def separate_cm_data(self):
        return "cm" in getattr(self, "public_config", {}).get("data", {})

    def set_data(self, dataloader_train, dataloader_val=None):
        super().set_data(dataloader_train, dataloader_val)
        if self.model_config["name"] == "wan_t2v" and not hasattr(dataloader_train.sampler, "configure_from_microbatch_offset"):
            from torch.utils.data import DataLoader

            from .cursor import TurboCursorSampler

            split = self.public_config["data"]["train"]
            sampler = TurboCursorSampler(
                dataloader_train.dataset,
                seed=self.config.get("seed", 42),
                shuffle=split.get("shuffle", True),
                num_replicas=get_data_parallel_world_size(),
                rank=get_data_parallel_rank(),
            )
            self.dataloader_train = DataLoader(
                dataloader_train.dataset,
                batch_size=1,
                sampler=sampler,
                num_workers=dataloader_train.num_workers,
                pin_memory=dataloader_train.pin_memory,
                collate_fn=dataloader_train.collate_fn,
            )
        if self.separate_cm_data:
            from lightx2v_train.data.utils import require_singleton_dataloader
            from lightx2v_train.utils.registry import build_data

            self.dataloader_cm = build_data(self.public_config, train_or_val="cm")
            require_singleton_dataloader(self.dataloader_cm, "Turbo CM dataloader")

    def _recipe(self):
        training = {key: value for key, value in self.public_config["training"].items() if key not in {"output_dir", "save_every_iters", "save_total_limit"}}
        data = {key: value for key, value in self.public_config["data"]["train"].items() if key not in {"num_workers", "pin_memory"}}
        value = {
            "upstream": core.UPSTREAM_REVISION,
            "seed": self.public_config.get("seed", 42),
            "model": self.public_config["model"],
            "training": training,
            "data": data,
            "scheduler": self.public_config["scheduler"],
            "distributed": self.public_config.get("distributed", {}),
            "dataset": self._dataset_fingerprint(),
        }
        if self.separate_cm_data:
            value["cm_data"] = {key: item for key, item in self.public_config["data"]["cm"].items() if key not in {"num_workers", "pin_memory"}}
            value["cm_dataset"] = self._dataset_fingerprint("cm")
        # JSON normalizes tuple geometry and integer dictionary keys.
        return json.loads(json.dumps(value, sort_keys=True))

    def _dataset_fingerprint(self, stream="train"):
        attribute = "_turbo_dataset_fingerprint" if stream == "train" else "_turbo_cm_dataset_fingerprint"
        if not hasattr(self, attribute):
            dataset = (self.dataloader_train if stream == "train" else self.dataloader_cm).dataset
            if not hasattr(dataset, "samples"):
                raise ValueError("Turbo requires indexed metadata samples for strict data resume")
            digest = hashlib.sha256()
            for record in dataset.samples:
                digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))
                digest.update(b"\n")
            setattr(self, attribute, {"rows": len(dataset.samples), "sha256": digest.hexdigest()})
        return getattr(self, attribute)

    def _expected_cursors(self, iteration):
        if self.separate_cm_data:
            return {
                "consumed_microbatches": self.turbo.consumed_dmd_microbatches(iteration, self.gradient_accumulation_iters),
                "consumed_cm_microbatches": self.turbo.consumed_cm_microbatches(iteration, self.gradient_accumulation_iters),
            }
        return {"consumed_microbatches": self.turbo.consumed_microbatches(iteration, self.gradient_accumulation_iters)}

    def _checkpoint_common(self, iteration):
        return {"version": 2 if self.separate_cm_data else self.checkpoint_version, "iteration": iteration, "world_size": get_world_size()}

    @staticmethod
    def _rank():
        return dist.get_rank() if dist.is_initialized() else 0

    def _resume_cursor(self, checkpoint, iteration):
        if checkpoint is None:
            return
        directory = Path(checkpoint)
        marker = directory / "turbo.complete.json"
        if not marker.is_file():
            raise RuntimeError(f"Not a complete Turbo checkpoint: {directory}; use a fresh output directory")
        expected = self._checkpoint_common(iteration)
        if json.loads(marker.read_text()) != expected:
            raise RuntimeError(f"Turbo completion metadata mismatch: {directory}")
        path = directory / f"turbo.rank{self._rank():05d}.json"
        state = json.loads(path.read_text())
        expected.update(
            rank=self._rank(),
            recipe=self._recipe(),
            fake_updates=self.turbo.fake_updates_before(iteration),
            **self._expected_cursors(iteration),
        )
        if state != expected:
            raise RuntimeError(f"Turbo recipe, counters or data cursor mismatch: {path}")
        self.fake_updates = state["fake_updates"]
        self.consumed_microbatches = state["consumed_microbatches"]
        self.consumed_cm_microbatches = state.get("consumed_cm_microbatches", 0)

    def _resolve_turbo_resume(self):
        directory = Path(self.output_train_dir)
        candidates = sorted(directory.glob("checkpoint-[0-9]*"))
        if not self.auto_resume:
            if candidates:
                raise RuntimeError("Turbo with auto_resume=false requires a fresh output directory")
            return None, 0
        complete = [path for path in candidates if (path / "turbo.complete.json").is_file()]
        if candidates and not complete:
            raise RuntimeError(f"No complete Turbo checkpoint in {directory}; PDMD and Turbo require separate outputs")
        if not complete:
            return None, 0
        path = complete[-1]
        iteration = int(path.name.split("-")[-1])
        logger.info("[turbo] resuming complete checkpoint {} student_updates={}", path, iteration)
        return str(path), iteration

    def setup(self, resume_ckpt_path=None):
        # Build the same resources on every stage and resume boundary. The
        # fake network receives updates only in the joint stage.
        super().setup(resume_ckpt_path=None)
        self.fake_lr_scheduler = self._build_lr_scheduler(
            self.fake_optimizer,
            num_warmup_steps=0,
            num_training_steps=max(1, self.turbo.joint_steps * self.turbo.fake_update_ratio),
        )
        if resume_ckpt_path is not None:
            self._load_resume_state(resume_ckpt_path)
        for capability in (self.student, self.fake, self.teacher):
            capability.set_training(False)
        if getattr(self, "model_config", {}).get("name") == "wan_t2v":
            components = {name: role.denoiser() for name, role in (("student", self.student), ("fake", self.fake), ("teacher", self.teacher))}
            for name, wrapper in (("vae", self.model.vae), ("t5", self.model.text_encoder)):
                if wrapper is not None:
                    components[name] = wrapper.model
            for name, module in components.items():
                parameters = list(module.parameters())
                dtypes = {parameter.dtype for parameter in parameters if parameter.is_floating_point()}
                devices = {parameter.device for parameter in parameters}
                if dtypes != {torch.float32} or devices != {self.model.device}:
                    raise ValueError(f"Wan Turbo requires resident FP32 {name}; got devices={devices}, dtypes={dtypes}")
                logger.info("[turbo] resident component={} device={} dtype=fp32 parameters={}", name, self.model.device, sum(parameter.numel() for parameter in parameters))

    def run_inference(self, current_iter):
        try:
            return super().run_inference(current_iter)
        finally:
            # The shared inference runtime restores trainable parameters and
            # training mode. Turbo's deterministic targets use eval mode.
            self.student.set_training(False)

    def save_checkpoint(self, iteration, save_total_limit):
        # Only save at completed student+fake groups: no half-finished joint
        # update or trajectory needs to be reconstructed during resume.
        # The generic DMD marker is written before this method's strict cursor
        # receipt. Keep old recoverable Turbo checkpoints until the new receipt
        # and per-rank states have all completed.
        if self.fake_updates != self.turbo.fake_updates_before(iteration) or any(getattr(self, key) != value for key, value in self._expected_cursors(iteration).items()):
            raise RuntimeError("Cannot checkpoint an incomplete Turbo student/fake update group")
        super().save_checkpoint(iteration, None)
        directory = Path(self.output_train_dir) / f"checkpoint-{iteration:09d}"
        common = self._checkpoint_common(iteration)
        state = dict(common, rank=self._rank(), recipe=self._recipe(), fake_updates=self.fake_updates, **self._expected_cursors(iteration))
        target = directory / f"turbo.rank{self._rank():05d}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
        temporary.replace(target)
        barrier()
        if is_main_process():
            marker = directory / "turbo.complete.json"
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps(common, sort_keys=True) + "\n")
            temporary.replace(marker)
        barrier()
        if is_main_process() and save_total_limit is not None:
            complete = sorted(path for path in directory.parent.glob("checkpoint-[0-9]*") if (path / "turbo.complete.json").is_file())
            for old in complete[: -int(save_total_limit)]:
                shutil.rmtree(old)
        barrier()

    @contextmanager
    def _random_stream(self, role, iteration, substep, microstep):
        device = self.student.device
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        seed = stream_seed(self.config.get("seed", 42), "turbo", role, iteration, substep, microstep, self._rank())
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            for index in cuda_devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            yield

    def _times(self, *, score=False):
        normal = torch.randn((), device=self.student.device, dtype=torch.float64)
        mean = self.turbo.score_time_mean if score else self.turbo.generator_time_mean
        std = self.turbo.score_time_std if score else self.turbo.generator_time_std
        base = torch.sigmoid(normal * std + mean)
        return tuple(core.rf_to_trig_time(core.shift_rf_time(base, shift)) for shift in (self.turbo.video_shift, self.turbo.audio_shift))

    def _corrupt(self, clean, theta):
        epsilon = noise_like(clean)
        return joint_like(clean, *[core.trig_corrupt(component(clean, name), component(epsilon, name), theta[i]) for i, name in enumerate(modality_names(clean))])

    def _weighted(self, losses):
        return self.turbo.video_weight * losses[0] + (self.turbo.audio_weight * losses[1] if len(losses) > 1 else 0)

    def _training_condition(self, sample):
        condition, negative = self._encode_conditions(sample)
        if getattr(self, "guidance_scale", 1.0) > 1:
            if negative is None:
                raise ValueError("Turbo CFG teacher requires an unconditional condition")
            return TurboCondition(condition, negative)
        return condition

    def _teacher_prediction(self, state, theta, condition):
        prediction = predict_trig(self.teacher, state, theta, condition)
        if not isinstance(condition, TurboCondition):
            return prediction
        unconditional = predict_trig(self.teacher, state, theta, condition.negative)
        return tuple(core.TrigPrediction(*(u + self.guidance_scale * (c - u) for c, u in zip(conditional, uncond, strict=True))) for conditional, uncond in zip(prediction, unconditional, strict=True))

    @staticmethod
    def _loss_metrics(prefix, losses):
        return {f"{prefix}_{name}": loss.detach() for name, loss in zip(("video", "audio"), losses)}

    def _dcm_loss(self, clean, condition):
        names = modality_names(clean)
        draw = torch.rand((), device=self.student.device, dtype=torch.float64)
        grids = tuple(
            core.dcm_schedule(draw, total_steps=self.turbo.dcm_total_steps, skip_steps=self.turbo.dcm_skip_steps, shift=shift)
            for shift in (self.turbo.video_shift, self.turbo.audio_shift)[: len(names)]
        )
        theta = tuple(grid[0] for grid in grids)
        state = self._corrupt(clean, theta)
        # Evaluate the detached target first, releasing all target activations
        # before the one parameter-gradient student forward.
        with torch.no_grad():
            target_state = state
            for index in range(self.turbo.dcm_skip_steps):
                times = tuple(grid[index] for grid in grids)
                teacher = self._teacher_prediction(target_state, times, condition)
                target_state = joint_like(target_state, *[core.trig_euler_step(component(target_state, name), teacher[i].F, times[i], grids[i][index + 1]) for i, name in enumerate(names)])
            target = predict_trig(self.student, target_state, tuple(grid[-1] for grid in grids), condition)
        prediction = predict_trig(self.student, state, theta, condition)
        losses = tuple(core.dcm_loss(prediction[i].x0, target[i].x0, scale=self.turbo.loss_scale) for i in range(len(names)))
        return self._weighted(losses), self._loss_metrics("dcm", losses)

    @torch.no_grad()
    def _trig_jvp(self, state, theta, teacher, condition):
        names = modality_names(state)
        tangent = tuple(core.scm_tangent(component(state, name), teacher[i].F, theta[i]) for i, name in enumerate(names))
        inputs = tuple(core.preprocess_trig_jvp(component(state, name), theta[i], tangent[i][0], tangent[i][1]) for i, name in enumerate(names))
        sigmas, sigma_tangent = torch.stack([item[1] for item in inputs]).float(), torch.stack([item[3] for item in inputs]).float()
        if torch.is_tensor(state):
            sigmas, sigma_tangent = sigmas[0], sigma_tangent[0]
        velocity, velocity_jvp = self.student.predict_velocity_jvp(
            joint_like(state, *[item[0] for item in inputs]),
            sigmas,
            condition.positive if isinstance(condition, TurboCondition) else condition,
            joint_like(state, *[item[2] for item in inputs]),
            sigma_tangent,
        )
        return tuple(
            core.postprocess_trig_jvp(component(state, name), theta[i], component(velocity, name), tangent[i][0], tangent[i][1], component(velocity_jvp, name))[1].F.detach()
            for i, name in enumerate(names)
        )

    def _scm_loss(self, clean, condition, iteration):
        theta = self._times()
        state = self._corrupt(clean, theta)
        with torch.no_grad():
            teacher = self._teacher_prediction(state, theta, condition)
            jvp = self._trig_jvp(state, theta, teacher, condition)
        prediction = predict_trig(self.student, state, theta, condition)
        elapsed = iteration - self.turbo.dcm_steps
        warmup = 1.0 if not self.turbo.tangent_warmup_steps else min(1.0, max(0, elapsed) / self.turbo.tangent_warmup_steps)
        directions = tuple(
            core.scm_direction(component(state, name), prediction[i].F, teacher[i].F, jvp[i], theta[i], warmup_ratio=warmup, norm_offset=self.turbo.scm_normalizer_epsilon)
            for i, name in enumerate(modality_names(state))
        )
        losses = tuple(core.scm_loss(prediction[i].F, directions[i], scale=self.turbo.loss_scale) for i in range(len(directions)))
        return self._weighted(losses), {**self._loss_metrics("scm", losses), "tangent_warmup": losses[0].new_tensor(warmup)}

    def _generate(self, reference, condition, *, with_grad):
        return sample_turbo_latents(
            self.student, noise_like(reference), condition, num_steps=self.turbo.num_inference_steps, video_shift=self.turbo.video_shift, audio_shift=self.turbo.audio_shift, with_grad=with_grad
        )

    def _dmd_loss(self, clean, condition):
        generated = self._generate(clean, condition, with_grad=True)
        theta = self._times(score=True)
        with torch.no_grad():
            query = self._corrupt(generated, theta)
            fake = predict_trig(self.fake, query, theta, condition)
            teacher = self._teacher_prediction(query, theta, condition)
        losses = tuple(
            core.dmd_loss(component(generated, name), fake[i].x0, teacher[i].x0, scale=self.turbo.loss_scale_dmd, normalization_floor=self.turbo.dmd_normalizer_floor)
            for i, name in enumerate(modality_names(generated))
        )
        return self._weighted(losses), self._loss_metrics("dmd", losses)

    def _critic_loss(self, clean, condition):
        with torch.no_grad():
            generated = self._generate(clean, condition, with_grad=False)
            theta = self._times(score=True)
            query = self._corrupt(generated, theta)
        prediction = predict_trig(self.fake, query, theta, condition)
        losses = tuple(core.critic_loss(component(generated, name), prediction[i].x0, theta[i]) for i, name in enumerate(modality_names(generated)))
        return self._weighted(losses), self._loss_metrics("critic", losses)

    def _noise_reference(self, sample):
        """Allocate only a geometry holder; DMD never needs a cached target."""
        from lightx2v_train.model_zoo.minimax_h3.capability_adapters.common import MiniMaxH3JointLatents

        shape = self._latent_shape(sample)
        if isinstance(shape, tuple):
            return torch.empty(shape, device=self.student.device, dtype=torch.float32)
        return MiniMaxH3JointLatents(
            torch.empty(shape.video_tokens, device=self.student.device, dtype=torch.float32),
            torch.empty(shape.audio_tokens, device=self.student.device, dtype=torch.float32),
            shape,
        )

    def _micro_objectives(self, samples, cm_samples, role, stage, iteration, substep, micro):
        """Yield closures so FSDP sync is configured BEFORE each forward.

        In split mode the CM and online objectives use their own conditions,
        geometry, deterministic random streams, cursor and route weight.
        Keeping the legacy branch preserves existing single-stream recipes.
        """
        student = role == "student"
        if not self.separate_cm_data:
            sample = next(samples)
            cursor = self.consumed_microbatches
            self.consumed_microbatches += 1
            scale = self.route_loss_scale(cursor) / self.gradient_accumulation_iters
            with self._random_stream(role, iteration, substep, micro):
                with torch.no_grad():
                    condition = self._training_condition(sample)
                    if not student and getattr(self, "model_config", {}).get("name") == "wan_t2v":
                        clean = self._noise_reference(sample)
                    else:
                        clean = self.student.extract_turbo_latents(sample, self._latent_shape(sample))
                if not student:
                    yield lambda: self._critic_loss(clean, condition), scale
                else:
                    yield (lambda: self._dcm_loss(clean, condition)) if stage == "dcm" else (lambda: self._scm_loss(clean, condition, iteration)), scale
                    if stage == "joint":
                        yield lambda: self._dmd_loss(clean, condition), scale
            return
        if student:
            if cm_samples is None:
                raise ValueError("Separate Turbo CM data requires a CM sample iterator")
            sample = next(cm_samples)
            cursor = self.consumed_cm_microbatches
            self.consumed_cm_microbatches += 1
            scale = self.cm_route_loss_scale(cursor) / self.gradient_accumulation_iters
            with self._random_stream("cm", iteration, 0, micro):
                with torch.no_grad():
                    condition = self._training_condition(sample)
                    clean = self.student.extract_turbo_latents(sample, self._latent_shape(sample))
                yield (lambda: self._dcm_loss(clean, condition)) if stage == "dcm" else (lambda: self._scm_loss(clean, condition, iteration)), scale
            # Release CM tensors before reading the independent online batch.
            del clean, condition, sample
        if not student or stage == "joint":
            sample = next(samples)
            cursor = self.consumed_microbatches
            self.consumed_microbatches += 1
            scale = self.route_loss_scale(cursor) / self.gradient_accumulation_iters
            with self._random_stream("student_dmd" if student else "fake", iteration, substep, micro):
                with torch.no_grad():
                    condition = self._training_condition(sample)
                    reference = self._noise_reference(sample)
                yield (lambda: self._dmd_loss(reference, condition)) if student else (lambda: self._critic_loss(reference, condition)), scale

    def _optimizer_step(self, samples, role, stage, iteration, substep=0, cm_samples=None):
        student = role == "student"
        optimizer = self.optimizer if student else self.fake_optimizer
        scheduler = self.lr_scheduler if student else self.fake_lr_scheduler
        parameters = self.trainable_params if student else self.fake_trainable_params
        sync = self._set_student_gradient_sync if student else self._set_fake_gradient_sync
        self.optimizer.zero_grad(set_to_none=True)
        self.fake_optimizer.zero_grad(set_to_none=True)
        totals = {}
        for micro in range(self.gradient_accumulation_iters):
            objective_count = 2 if student and stage == "joint" else 1
            for index, (objective, scale) in enumerate(self._micro_objectives(samples, cm_samples, role, stage, iteration, substep, micro)):
                sync(micro == self.gradient_accumulation_iters - 1 and index == objective_count - 1)
                loss, metrics = objective()
                (loss * scale).backward()
                for key, value in {"loss": loss.detach(), **metrics}.items():
                    totals[key] = totals.get(key, 0.0) + float(value) / self.gradient_accumulation_iters
                del loss
        self._sync_sequence_parallel_grads(parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        if hasattr(norm, "full_tensor"):
            norm = norm.full_tensor()
        finite = torch.isfinite(norm).to(device=self.student.device, dtype=torch.int32)
        if any(not math.isfinite(value) for value in totals.values()):
            finite.zero_()
        if dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError(f"Turbo nonfinite {role} loss or gradient at student update {iteration + 1}")
        optimizer.step()
        scheduler.step()
        if student:
            self._after_student_optimizer_step("main")
        else:
            self.fake_updates += 1
        totals["grad_norm"] = float(norm)
        return totals

    def _iter_cm_samples(self):
        epoch = 0
        while True:
            sampler = self.dataloader_cm.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            yield from self.dataloader_cm
            epoch += 1

    def train(self):
        checkpoint, iteration = self._resolve_turbo_resume()
        self._resume_cursor(checkpoint, iteration)
        sampler = getattr(self.dataloader_train, "sampler", None)
        if not hasattr(sampler, "configure_from_microbatch_offset"):
            raise ValueError("H3 Turbo requires a sampler with an explicit microbatch cursor")
        sampler.configure_from_microbatch_offset(
            start_microbatch=self.consumed_microbatches,
            gradient_accumulation_iters=self.gradient_accumulation_iters,
            fake_update_ratio=self.fake_update_ratio,
            data_consumption_mode="turbo_dmd_student_and_fake_fresh" if self.separate_cm_data else "turbo_student_and_fake_fresh",
            student_data_source="condition_only_online_rollout" if self.separate_cm_data else "cached_clean_av",
        )
        self.route_loss_scale = getattr(sampler, "cursor_microbatch_loss_scale", self.route_loss_scale)
        if self.separate_cm_data:
            cm_sampler = self.dataloader_cm.sampler
            if not hasattr(cm_sampler, "configure_from_microbatch_offset"):
                raise ValueError("Turbo CM data requires an absolute microbatch cursor sampler")
            cm_sampler.configure_from_microbatch_offset(
                start_microbatch=self.consumed_cm_microbatches,
                gradient_accumulation_iters=self.gradient_accumulation_iters,
                fake_update_ratio=self.fake_update_ratio,
            )
            self.cm_route_loss_scale = getattr(cm_sampler, "cursor_microbatch_loss_scale", lambda _: 1.0)
        self.setup(resume_ckpt_path=checkpoint)
        if getattr(self, "infer_every_iters", None):
            self.inferencer.set_data(self.dataloader_val)
            if iteration == 0 and getattr(self, "infer_config", {}).get("run_at_start", True):
                self.run_inference(0)
            elif checkpoint is not None and iteration == self.max_train_iters and iteration % self.infer_every_iters == 0:
                # Saving precedes previews. A completed checkpoint can survive
                # a failed final preview, so retry it before a staged launcher
                # treats this resumed stage as successfully finished.
                self.run_inference(iteration)
        if is_main_process():
            Path(self.output_train_dir).mkdir(parents=True, exist_ok=True)
        barrier()
        logger.info(
            "[turbo] student_updates={}/{} fake_updates={} curriculum={}/{}/{} global_batch={} target_steps={} shifts={}/{} JVP=exact source={}",
            iteration,
            self.max_train_iters,
            self.fake_updates,
            self.turbo.dcm_steps,
            self.turbo.scm_steps,
            self.turbo.joint_steps,
            get_world_size() * self.gradient_accumulation_iters,
            self.num_inference_steps,
            self.turbo.video_shift,
            self.turbo.audio_shift,
            core.UPSTREAM_REVISION,
        )
        samples = self._iter_train_samples()
        cm_samples = self._iter_cm_samples() if self.separate_cm_data else None
        last_saved = iteration if checkpoint else -1
        while iteration < self.max_train_iters:
            stage = self.turbo.stage_at(iteration)
            self.student.on_iteration_start(iteration)
            if self.separate_cm_data:
                metrics = self._optimizer_step(samples, "student", stage, iteration, cm_samples=cm_samples)
            else:
                metrics = self._optimizer_step(samples, "student", stage, iteration)
            if stage == "joint":
                for substep in range(self.turbo.fake_update_ratio):
                    fake_metrics = self._optimizer_step(samples, "fake", stage, iteration, substep)
                    for key, value in fake_metrics.items():
                        metrics[f"fake_{key}"] = metrics.get(f"fake_{key}", 0.0) + value / self.turbo.fake_update_ratio
            iteration += 1
            self.student.on_iteration_end(iteration)
            if iteration == 1 or iteration % self.train_log_every_iters == 0 or iteration == self.max_train_iters:
                metrics = {name: reduce_mean(value) for name, value in metrics.items()}
                logger.info(
                    "[turbo] student_updates={}/{} stage={} fake_updates={} optimizer_updates={} consumed_microbatches={} metrics={}",
                    iteration,
                    self.max_train_iters,
                    stage,
                    self.fake_updates,
                    iteration + self.fake_updates,
                    self.consumed_microbatches,
                    metrics,
                )
                self.log_metrics({f"train/turbo_{key}": value for key, value in metrics.items()}, step=iteration)
                if self.separate_cm_data:
                    logger.info("[turbo][data] cm_microbatches={} online_dmd_microbatches={}", self.consumed_cm_microbatches, self.consumed_microbatches)
            if self.save_every_iters and iteration % self.save_every_iters == 0:
                self.save_checkpoint(iteration, self.save_total_limit)
                last_saved = iteration
            if getattr(self, "infer_every_iters", None) and iteration % self.infer_every_iters == 0:
                self.run_inference(iteration)
        if iteration != last_saved:
            self.save_checkpoint(iteration, self.save_total_limit)
