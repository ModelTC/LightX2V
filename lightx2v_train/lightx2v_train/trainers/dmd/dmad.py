"""H3 DMAD, including reference-conditioned audio-video generation.

Algorithm adapted from Yzmblog/DMAD, Apache-2.0, commit
8067c05f74a8cfc818d6e21c2b49405b49ba9cbc (train/h3).
No teacher score model is built: teacher supervision is an offline sample.
"""

import copy
import math
import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.nn import functional as F

from lightx2v_train.model_capabilities import DistributionMatchingCapability, ParallelCapability, TrainableModelCapability
from lightx2v_train.model_zoo import build_loaded_model
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.common import MiniMaxH3JointLatents
from lightx2v_train.runtime.checkpoint import prune_checkpoints
from lightx2v_train.runtime.distributed import barrier, get_rank, get_world_size, is_main_process
from lightx2v_train.schedulers import DMDFlowMatchingScheduler
from lightx2v_train.utils.registry import TRAINER_REGISTER

from ..base import BaseTrainer
from .checkpoint import DmdCheckpointManager
from .dmad_math import DmadConfig, DualHeadCritic, GapRouter, PowerEMA, generator_loss, rollout_base_sigmas
from .trainer import DmdTrainer


class DmadCheckpointManager(DmdCheckpointManager):
    def _extra_checkpoint_metadata(self):
        return {**super()._extra_checkpoint_metadata(), "dmad": self.dmad_metadata()}

    def _validate_checkpoint_state(self, state, state_path, resume_ckpt_path):
        if state.get("dmad") != self.dmad_metadata():
            raise ValueError("DMAD checkpoint objective/data/rollout differs; use a fresh output directory.")
        self._require_checkpoint_keys(state, ["dmad_heads", "dmad_head_optimizer", "dmad_gap", "dmad_ema"], state_path)
        super()._validate_checkpoint_state(state, state_path, resume_ckpt_path)

    def _extra_residual_head_training_state(self):
        # The parent calls this extension in both distributed and local saves.
        sharded = self.parallel.is_fsdp() or self._parallel(self.fake_model).is_fsdp()
        ema = [bank.state_dict() for bank in self.dmad_emas]
        if sharded:
            ema = [{k: v for k, v in item.items() if k != "shadow"} for item in ema]
        return {
            "dmad_heads": self.critic_heads.state_dict(),
            "dmad_head_optimizer": self.head_optimizer.state_dict(),
            "dmad_gap": self.gap_router.state_dict(),
            "dmad_ema": ema,
        }

    def _load_residual_head_training_state(self, state):
        self.critic_heads.load_state_dict(state["dmad_heads"])
        self.head_optimizer.load_state_dict(state["dmad_head_optimizer"])
        self.gap_router.load_state_dict(state["dmad_gap"])
        self.owner._saved_dmad_ema = state["dmad_ema"]
        if len(self._saved_dmad_ema) != len(self.dmad_emas):
            raise ValueError("DMAD EMA bank count changed.")
        for bank, saved in zip(self.dmad_emas, self._saved_dmad_ema):
            if "shadow" in saved:
                bank.load_state_dict(saved)

    def _save_distributed_state(self, save_dir, iteration):
        super()._save_distributed_state(save_dir, iteration)
        if self.dmad_emas:
            dcp.save({f"ema_{i}": bank.shadow for i, bank in enumerate(self.dmad_emas)}, checkpoint_id=os.path.join(save_dir, "dmad_ema"))

    def _load_distributed_state(self, resume_ckpt_path):
        super()._load_distributed_state(resume_ckpt_path)
        if self.dmad_emas:
            state = {f"ema_{i}": bank.shadow for i, bank in enumerate(self.dmad_emas)}
            dcp.load(state, checkpoint_id=os.path.join(resume_ckpt_path, "dmad_ema"))
            for i, (bank, saved) in enumerate(zip(self.dmad_emas, self._saved_dmad_ema)):
                bank.load_state_dict({**saved, "shadow": state[f"ema_{i}"]})


@TRAINER_REGISTER("dmad")
class DmadTrainer(DmdTrainer):
    trainer_name = "dmad"
    supports_diversity_loss = False
    supports_real_data_fake = False
    supports_ida = False

    def __init__(self, config):
        super().__init__(config)
        # The shared DMD parser defaults CFG to 3. H3 DMAD is guidance-distilled
        # and deliberately has no online teacher configuration/model.
        if "guidance_scale" not in self.training_config.get("teacher", {}):
            self.guidance_scale = 1.0
        self.dmad_config = DmadConfig.from_mapping(self.training_config.get("dmad"))
        self.checkpoint_manager = DmadCheckpointManager(self)
        if self.model_config.get("name") not in {"minimax_h3_ref2av", "minimax_h3_t2av"}:
            raise ValueError("DMAD currently supports only MiniMax H3 T2AV/Ref2AV.")
        if self.residual_head_config.enabled or self.student_ema_config["enabled"]:
            raise ValueError("DMAD uses its own heads and power EMA, not DMD residual_head/student.ema.")
        if int(config.get("distributed", {}).get("sequence_parallel", {}).get("size", 1)) != 1:
            raise ValueError("DMAD target-feature training currently requires sequence_parallel.size=1.")
        if self.dmd_update_order != "student_first" or self.random_schedule_enabled:
            raise ValueError("DMAD uses student_first and its own random re-noise rollout.")
        if self.infer_every_iters:
            raise ValueError("Use inference.method=none for training; DMAD sampling must use re-noise, not the DMD Euler inferencer.")
        if self.student_checkpoint_path:
            raise ValueError("DMAD warm-start checkpoint_path is not supported; use a complete DMAD resume or a fresh run.")
        if self.model_config.get("capabilities", {}).get("distribution_matching", {}).get("projected_dmd", False):
            raise ValueError("DMAD is not projected DMD; set projected_dmd=false.")
        if not self.dmad_config.gap_sync and get_world_size() > 1:
            raise ValueError("Distributed DMAD requires gap_sync=true for a shared resumable routing state.")
        self.max_grad_norm = float(self.max_grad_norm)
        if self.max_grad_norm < 0 or not math.isfinite(self.max_grad_norm):
            raise ValueError("max_grad_norm must be finite and nonnegative (0 disables clipping).")
        self._resume_rng = None

    def setup(self, resume_ckpt_path=None):
        # Deliberately bypass _DmdRuntime.setup: it would allocate an unused
        # third H3 transformer, causing avoidable GPU/host-memory pressure.
        BaseTrainer.setup(self, resume_ckpt_path=None)
        base = {key: copy.deepcopy(value) for key, value in self.model_config.items() if key not in {"fake", "teacher", "student"}}
        self.fake_model_config = self._build_dmd_role_config("fake", base)
        self.fake_model = build_loaded_model(self.fake_model_config, load_transformer=True, load_vae=False, load_condition_encoder=False)
        self.fake_model.reuse_frozen_components_from(self.model)
        self.fake = self.fake_model.capabilities.require(DistributionMatchingCapability)
        self._setup_trainable_model(self.fake_model, role="fake")
        critic = self.fake.denoiser()
        block = self.dmad_config.feature_block
        if block >= len(critic.transformer_blocks):
            raise ValueError(f"DMAD feature_block={block} outside the H3 backbone.")
        # These modules are not traversed by the feature forward. Do not leave
        # unused optimizer slots/adapters whose missing gradients break resume.
        for module in list(critic.transformer_blocks[block + 1 :]) + [getattr(critic, key, None) for key in ("norm_out", "proj_out", "audio_proj_out")]:
            if module is not None:
                module.requires_grad_(False)
        self.fake_model.capabilities.require(ParallelCapability).apply(self.fake_model_config)
        if self.gradient_checkpointing:
            self.fake_model.capabilities.require(TrainableModelCapability).enable_gradient_checkpointing()
        self.fake_trainable_params = [p for p in self.fake_model.capabilities.require(TrainableModelCapability).parameters() if p.requires_grad]
        self.fake_optimizer = self._build_optimizer(self.fake_trainable_params, self.fake_optimizer_config)
        self.fake_lr_scheduler = self._build_lr_scheduler(self.fake_optimizer, num_warmup_steps=0, num_training_steps=self.max_train_iters * self.fake_update_ratio)
        self.teacher_model = self.teacher = self.fake_real_model = None
        self.scheduler = DMDFlowMatchingScheduler(self.config)
        self.critic_heads = DualHeadCritic(int(critic.config.hidden_size)).to(device=self.student.device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            for value in self.critic_heads.state_dict().values():
                dist.broadcast(value, src=0)
        self.head_optimizer = self._build_optimizer(list(self.critic_heads.parameters()), self.fake_optimizer_config)
        self.gap_router = GapRouter(self.dmad_config).to(self.student.device)
        self.dmad_emas = [PowerEMA(self.parallel.state_module(), gamma) for gamma in self.dmad_config.ema_gammas]
        if resume_ckpt_path is not None:
            self._load_resume_state(resume_ckpt_path)
        logger.info(
            "[dmad] student={} critic={} no_online_teacher=true rollout=stochastic_renoise feature_block={} config={}",
            self.student_train_type,
            self.fake_train_type,
            block,
            self.dmad_config.metadata(),
        )

    def dmad_metadata(self):
        dataset = getattr(getattr(self, "dataloader_train", None), "dataset", None)
        return {
            "version": 1,
            "options": self.dmad_config.metadata(),
            "rollout": "random_running_min_renoise",
            "steps": self.num_inference_steps,
            "video_shift": self.student.video_shift,
            "audio_shift": self.student.audio_shift,
            "fake_update_ratio": self.fake_update_ratio,
            "gradient_accumulation_iters": self.gradient_accumulation_iters,
            "data_digest": getattr(dataset, "manifest_digest", None),
        }

    @contextmanager
    def _frozen_critic(self):
        params = [*self.fake_trainable_params, *self.critic_heads.parameters()]
        previous = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(False)
        self.fake.set_training(False)
        try:
            yield
        finally:
            for p, enabled in zip(params, previous):
                p.requires_grad_(enabled)

    def _critic_logits(self, latents, sigma, condition):
        features = self.fake.predict_dmad_features(latents, sigma, condition, self.dmad_config.feature_block)
        return self.critic_heads(features["video"], features["audio"])

    def _noise(self, latents, sigma):
        noise = self.student.random_noise_like(latents, torch.float32, lambda x: x)
        return self.student.add_noise(None, latents, noise, sigma)

    def run_dmad_rollout(self, condition, latent_shape, *, grad_enabled, initial_noise=None, random_timesteps=True):
        steps = self._sample_synced_int(1, self.num_inference_steps + 1) if random_timesteps else self.num_inference_steps
        sigmas = rollout_base_sigmas(steps, self.dmad_config.renoise_sigma_min, self.dmad_config.renoise_sigma_max, device=self.student.device, random_timesteps=random_timesteps)
        xt = self.sample_initial_latents(latent_shape) if initial_noise is None else initial_noise
        self.student.set_training(grad_enabled)
        for index in range(steps):
            context = torch.enable_grad if grad_enabled and index == steps - 1 else torch.no_grad
            with context():
                velocity = self.student.predict_velocity(xt, sigmas[index], condition)
                x0 = self.student.x0_from_velocity(xt, velocity, sigmas[index])
            if index + 1 < steps:
                with torch.no_grad():
                    xt = self._noise(self.student.detach(x0), sigmas[index + 1])
        return x0

    def _targets(self, sample, shape, field):
        data = sample[field]
        video = data["video"].to(device=self.student.device, dtype=torch.float32)
        audio = data["audio"].to(device=self.student.device, dtype=torch.float32)
        if tuple(video.shape) != shape.video_tokens or tuple(audio.shape) != shape.audio_tokens:
            raise ValueError(f"DMAD {field} target shapes {tuple(video.shape)}/{tuple(audio.shape)} do not match {shape}.")
        return MiniMaxH3JointLatents(video, audio, shape)

    @staticmethod
    def _check_finite(loss):
        valid = torch.isfinite(loss.detach()).all().to(torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not bool(valid):
            raise FloatingPointError("DMAD non-finite loss on at least one rank; optimizer not stepped.")

    def _train_one_stage(self, samples, stage, grad_accum_iters, outer_iteration=0, fake_update_index=0):
        del outer_iteration, fake_update_index
        if stage not in {"student", "fake"}:
            raise ValueError(f"Unknown DMAD stage {stage}")
        is_student = stage == "student"
        optimizer = self.optimizer if is_student else self.fake_optimizer
        schedule = self.lr_scheduler if is_student else self.fake_lr_scheduler
        params = self.trainable_params if is_student else self.fake_trainable_params
        optimizer.zero_grad(set_to_none=True)
        self.head_optimizer.zero_grad(set_to_none=True)
        metrics = {"loss": 0.0, "dmd": 0.0, "div_loss": 0.0, "real_dmd": 0.0, "fake_real": 0.0}
        for micro in range(grad_accum_iters):
            sample = next(samples)
            condition, _ = self._encode_conditions(sample)
            shape = self._latent_shape(sample)
            sync = self._set_student_gradient_sync if is_student else self._set_fake_gradient_sync
            sync(micro == grad_accum_iters - 1)
            sigma = torch.empty((), device=self.student.device).uniform_(self.dmad_config.renoise_sigma_min, self.dmad_config.renoise_sigma_max)
            band = self.gap_router.band(sigma)
            generated = self.run_dmad_rollout(condition, shape, grad_enabled=is_student)
            generated = self._noise(generated, sigma)
            if is_student:
                with self._frozen_critic():
                    gr, gt = self._critic_logits(generated, sigma, condition)
                    weight = self.gap_router.weight(band)
                    loss = generator_loss(gr, gt, weight, self.dmad_config.lambda_real, self.dmad_config.lambda_teacher)
                    self._check_finite(loss)
                    (loss / grad_accum_iters).backward()
                metrics["dmad_teacher_weight"] = metrics.get("dmad_teacher_weight", 0.0) + float(weight) / grad_accum_iters
                value = float(loss.detach())
                metrics["dmd"] += value / grad_accum_iters  # inherited logger key, not a score loss
            else:
                self.fake.set_training(True)
                # Sequential backward keeps only one critic activation graph
                # alive. All three sources share sigma AND the same refs/prompt.
                gr, gt = self._critic_logits(generated, sigma, condition)
                loss_g = (F.softplus(gr) + F.softplus(gt)).mean()
                self._check_finite(loss_g)
                (loss_g / grad_accum_iters).backward()
                value = float(loss_g.detach())
                del gr, gt, loss_g, generated
                real = self._noise(self._targets(sample, shape, "dmad_real"), sigma)
                rr, _ = self._critic_logits(real, sigma, condition)
                loss_r = F.softplus(-rr).mean()
                self._check_finite(loss_r)
                (loss_r / grad_accum_iters).backward()
                real_score = rr.detach()
                value += float(loss_r.detach())
                del rr, loss_r, real, _
                teacher = self._noise(self._targets(sample, shape, "dmad_teacher"), sigma)
                tr, tt = self._critic_logits(teacher, sigma, condition)
                loss_t = F.softplus(-tt).mean()
                self._check_finite(loss_t)
                (loss_t / grad_accum_iters).backward()
                self.gap_router.update(band, real_score, tr)
                value += float(loss_t.detach())
                del tr, tt, teacher, loss_t
            metrics["loss"] += value / grad_accum_iters

        if not is_student:
            for p in self.critic_heads.parameters():
                if p.grad is not None and dist.is_available() and dist.is_initialized():
                    dist.all_reduce(p.grad)
                    p.grad.div_(get_world_size())
        limit = self.max_grad_norm or float("inf")
        norm = torch.nn.utils.clip_grad_norm_(params, limit)
        if hasattr(norm, "full_tensor"):
            norm = norm.full_tensor()
        self._check_finite(norm)
        if not is_student:
            self._check_finite(torch.nn.utils.clip_grad_norm_(self.critic_heads.parameters(), limit))
        optimizer.step()
        if is_student:
            for bank in self.dmad_emas:
                bank.update()
        else:
            self.head_optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        self.head_optimizer.zero_grad(set_to_none=True)
        return metrics

    def _iter_train_samples(self):
        # Creating a DataLoader iterator draws its worker seed. Restore model
        # RNG after that draw and before the first resumed rollout.
        iterator = super()._iter_train_samples()
        first = next(iterator)
        if self._resume_rng is not None:
            state = self._resume_rng
            torch.set_rng_state(state["torch"])
            random.setstate(state["python"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(state["cuda"], self.student.device)
            self._resume_rng = None
        yield first
        yield from iterator

    def _load_resume_state(self, path):
        if not (Path(path) / "dmad_complete").is_file():
            raise ValueError("Not a complete DMAD checkpoint; use a fresh DMAD output directory.")
        super()._load_resume_state(path)
        self._resume_rng = torch.load(Path(path) / f"rng_rank{get_rank()}.pt", map_location="cpu", weights_only=False)

    def save_checkpoint(self, iteration, save_total_limit):
        # Publish only after every model/head/EMA/rank state has been saved.
        output = self.output_train_dir
        os.makedirs(output, exist_ok=True)
        staging = [tempfile.mkdtemp(prefix=".dmad-save-", dir=output) if is_main_process() else None]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(staging, src=0)
        try:
            self.output_train_dir = staging[0]
            super().save_checkpoint(iteration, save_total_limit)
            checkpoint = Path(staging[0]) / f"checkpoint-{iteration:09d}"
            rng = {"torch": torch.get_rng_state(), "python": random.getstate()}
            if torch.cuda.is_available():
                rng["cuda"] = torch.cuda.get_rng_state(self.student.device)
            torch.save(rng, checkpoint / f"rng_rank{get_rank()}.pt")
            for i, bank in enumerate(self.dmad_emas):
                with bank.average_parameters():
                    self._save_model_weights(self.model, str(checkpoint / f"ema{i + 1}_student"), role="student")
            barrier()
            if is_main_process():
                (checkpoint / "dmad_complete").write_text("1\n", encoding="utf-8")
                final = Path(output) / checkpoint.name
                if final.exists():
                    raise FileExistsError(f"Refusing to replace DMAD checkpoint {final}")
                os.replace(checkpoint, final)
                os.rmdir(staging[0])
                # Prune only after the replacement checkpoint is complete.
                prune_checkpoints(output, save_total_limit + 1 if save_total_limit else save_total_limit)
            barrier()
        finally:
            self.output_train_dir = output
