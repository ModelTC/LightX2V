import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.fsdp import FSDPModule

from lightx2v_train.model_capabilities import DistributionMatchingCapability, ParallelCapability
from lightx2v_train.model_zoo import build_loaded_model
from lightx2v_train.runtime.distributed import barrier, get_rank, get_world_size, is_main_process, reduce_mean
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.utils.registry import TRAINER_REGISTER

from ..base import BaseTrainer
from ..latent_ops import fork_rng, seed_rng

_MODALITIES = ("video", "audio")


def dcm_times(draw, shifts, total_steps, skip_steps):
    steps = torch.arange(skip_steps + 1, device=draw.device, dtype=draw.dtype)
    base = 1 - draw * (1 - skip_steps / total_steps) - steps / total_steps
    return tuple(torch.atan2(shift * base, 1 - base) for shift in shifts)


def _trig_prediction(model, state, times, condition):
    cos = times.cos()
    sin = times.sin()
    scale = cos + sin
    rf_state = type(state)(
        state.video / scale[0].reshape((1,) * state.video.ndim),
        state.audio / scale[1].reshape((1,) * state.audio.ndim),
        state.shape,
    )
    velocity = model.predict_velocity_at_sigmas(model.to_dtype(rf_state, torch.float32), (sin / scale).float(), condition)
    x0, flow = [], []
    for i, name in enumerate(_MODALITIES):
        x, v = getattr(state, name), getattr(velocity, name)
        c, s, total = (value[i].reshape((1,) * x.ndim) for value in (cos, sin, scale))
        x0.append((x + s * v) / total)
        flow.append(((c - s) * x - v) / total)
    return x0, flow


def dcm_loss(student, teacher, clean, condition, times, generator, weights, scale):
    theta = times[:, 0]
    parts = []
    for time, x in zip(theta, (clean.video, clean.audio), strict=True):
        time = time.reshape((1,) * x.ndim)
        noise = torch.randn(x.shape, dtype=torch.float32, device=x.device, generator=generator)
        parts.append(time.cos() * x + time.sin() * noise)
    state = type(clean)(*parts, clean.shape)
    with torch.no_grad():
        target = state
        for step in range(times.shape[1] - 1):
            _, flow = _trig_prediction(teacher, target, times[:, step], condition)
            dt = times[:, step] - times[:, step + 1]
            target = type(state)(target.video - dt[0] * flow[0], target.audio - dt[1] * flow[1], state.shape)
        target_x0, _ = _trig_prediction(student, target, times[:, -1], condition)
    x0, _ = _trig_prediction(student, state, theta, condition)
    losses = [(value - target).square().flatten(1).sum(1).mean() * scale for value, target in zip(x0, target_x0, strict=True)]
    return sum(weight * loss for weight, loss in zip(weights, losses, strict=True))


class AdapterEMA:
    def __init__(self, module, decay):
        self.module = module
        self.decay = decay
        self.weights = {name: param.detach().float().clone() for name, param in self.parameters().items() if param.requires_grad}

    def parameters(self):
        for layer in self.module.modules():
            if isinstance(layer, FSDPModule):
                layer.reshard()
        return dict(self.module.named_parameters())

    @torch.no_grad()
    def update(self):
        params = self.parameters()
        for name, weight in self.weights.items():
            weight.lerp_(params[name].detach().float(), 1 - self.decay)

    @contextmanager
    @torch.no_grad()
    def average_parameters(self):
        params = self.parameters()
        saved = {name: params[name].detach().clone() for name in self.weights}
        try:
            for name, weight in self.weights.items():
                params[name].copy_(weight)
            yield
        finally:
            for name, param in saved.items():
                params[name].copy_(param)


@TRAINER_REGISTER("dcm")
class DCMTrainer(BaseTrainer):
    trainer_name = "dcm"
    required_capabilities = (*BaseTrainer.required_capabilities, DistributionMatchingCapability)

    def __init__(self, config):
        super().__init__(config)
        self.dcm = self.training_config["dcm"]
        self.shifts = self.dcm["video_shift"], self.dcm["audio_shift"]
        self.total_steps = self.dcm.get("total_steps", 48)
        self.skip_steps = self.dcm.get("skip_steps", 1)
        self.loss_scale = self.dcm.get("loss_scale", 100.0)
        self.ema_decay = self.dcm.get("ema_decay", 0.999)
        self.loss_weights = self.dcm.get("video_loss_weight", 0.8), self.dcm.get("audio_loss_weight", 0.8)
        if not 1 <= self.skip_steps <= self.total_steps or min(self.shifts) <= 0:
            raise ValueError("dCM needs positive shifts and 1 <= skip_steps <= total_steps.")
        if self.train_type != "lora" or not 0 <= self.ema_decay <= 1:
            raise ValueError("H3 dCM needs LoRA training and an EMA decay in [0,1].")

    def _setup_trainable_model(self, model):
        self.trainable_model.configure(self.train_type, self.training_config["lora"])

    def setup(self, resume_ckpt_path=None):
        super().setup(resume_ckpt_path=resume_ckpt_path)
        self.student = self.model.capabilities.require(DistributionMatchingCapability)
        self.student.set_training(False)
        self.ema = AdapterEMA(self.parallel.state_module(), self.ema_decay)
        if resume_ckpt_path:
            self._load_ema(Path(resume_ckpt_path))

        teacher_config = deepcopy(self.config)
        teacher_config["model"].update(self.model_config.get("teacher", {}))
        self.teacher_model = build_loaded_model(teacher_config, load_transformer=True, load_vae=False, load_condition_encoder=False)
        self.teacher = self.teacher_model.capabilities.require(DistributionMatchingCapability)
        self.teacher.denoiser().requires_grad_(False)
        self.teacher_model.capabilities.require(ParallelCapability).apply(teacher_config)
        self.teacher_model.load_deferred_transformer_weights()
        self.teacher.set_training(False)

    def _load_ema(self, directory):
        if json.loads((directory / "dcm.json").read_text()) != self._recipe():
            raise ValueError(f"dCM checkpoint dataset or recipe changed: {directory}")
        if self.parallel.is_fsdp():
            dcp.load({"ema": self.ema.weights}, checkpoint_id=directory / "ema_state")
        else:
            saved = torch.load(directory / "ema_state.pt", map_location="cpu", weights_only=True)
            for name, weight in self.ema.weights.items():
                weight.copy_(saved[name])

    def _recipe(self):
        sampler = self.dataloader_train.sampler
        return {
            "dataset": self.dataloader_train.dataset.fingerprint,
            "sampler": [sampler.seed, sampler.shuffle, sampler.drop_last],
            "dcm": self.dcm,
            "accumulation": self.gradient_accumulation_iters,
        }

    def save_checkpoint(self, iteration, save_total_limit):
        directory = Path(self.output_train_dir) / f"checkpoint-{iteration:09d}"
        if is_main_process():
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "dcm.json").write_text(json.dumps(self._recipe()) + "\n")
        barrier()
        if self.parallel.is_fsdp():
            dcp.save({"ema": self.ema.weights}, checkpoint_id=directory / "ema_state")
        elif is_main_process():
            torch.save(self.ema.weights, directory / "ema_state.pt")
        with self.ema.average_parameters():
            self.checkpoint.save_weights(directory / "student_ema", self.train_type)
        barrier()
        super().save_checkpoint(iteration, save_total_limit)

    def _samples(self, offset):
        sampler = self.dataloader_train.sampler
        sampler.set_start(offset)
        epoch = 0
        while True:
            sampler.set_epoch(epoch)
            yield from self.dataloader_train
            epoch += 1

    def train(self):
        checkpoint, start = self._resolve_resume()
        if start >= self.max_train_iters:
            return
        self.setup(resume_ckpt_path=checkpoint)
        accum = self.gradient_accumulation_iters
        samples = self._samples(start * accum)
        broadcast = broadcast_sequence_parallel_value
        device = self.student.device
        generator = torch.Generator(device=device)

        for iteration in range(start, self.max_train_iters):
            self.optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for micro in range(accum):
                sample = next(samples)
                seed = self.config.get("seed", 42) + (iteration * accum + micro) * get_world_size() + get_rank()
                generator.manual_seed(seed)
                with fork_rng(device) as devices:
                    seed_rng(seed + 2**32, devices)
                    condition, _ = self.student.encode_conditions(sample, None, 1.0, broadcast)
                clean = self.student.extract_real_latents(sample, torch.float32, broadcast)
                draw = torch.rand((), device=device, dtype=torch.float64, generator=generator)
                times = torch.stack(dcm_times(draw, self.shifts, self.total_steps, self.skip_steps))
                self._set_gradient_sync(micro == accum - 1)
                loss = dcm_loss(self.student, self.teacher, clean, condition, times, generator, self.loss_weights, self.loss_scale) / accum
                loss.backward()
                loss_sum += loss.detach().item()
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.ema.update()
            done = iteration + 1
            if done == 1 or done % self.train_log_every_iters == 0 or done == self.max_train_iters:
                value = reduce_mean(loss_sum)
                logger.info("[dcm] update={}/{} loss={:.6f}", done, self.max_train_iters, value)
                self.log_metrics({"train/loss": value}, step=done)
            if done == self.max_train_iters or (self.save_every_iters and done % self.save_every_iters == 0):
                self.save_checkpoint(done, self.save_total_limit)
        self.finish_monitor()
