"""H3 PDD with student-only optimization and checkpointed staggered rollouts.

Algorithm reference: NVlabs/FastGen@898e757836dc244192533fe6ea797906799dd745,
methods/trajectory_matching/pdd.py and methods/backward_simulation.py.
The backbone uses this repository's LoRA/FSDP runtime; new output heads are
trained in full. This is a LoRA adaptation, not the upstream full-training recipe.
"""

import copy
import hashlib
import json
import shutil
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import DataLoader

from lightx2v_train.model_capabilities import DistributionMatchingCapability, ParallelCapability
from lightx2v_train.model_zoo import build_loaded_model
from lightx2v_train.runtime.distributed import (
    barrier,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_rank,
    get_sequence_parallel_world_size,
    get_world_size,
    is_main_process,
    reduce_mean,
)
from lightx2v_train.utils.registry import TRAINER_REGISTER

from ..base import BaseTrainer
from ..dmd.official_pdmd_core import stream_seed
from ..turbo.cursor import TurboCursorSampler
from .config import PDDConfig
from .core import advance_blocks, pdd_loss_from_prediction


def map_tensors(value, function):
    if torch.is_tensor(value):
        return function(value)
    if is_dataclass(value):
        return replace(value, **{field.name: map_tensors(getattr(value, field.name), function) for field in fields(value)})
    if isinstance(value, dict):
        return {key: map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(map_tensors(item, function) for item in value)
    return value


@TRAINER_REGISTER("pdd")
class PDDTrainer(BaseTrainer):
    trainer_name = "pdd"
    required_capabilities = (*BaseTrainer.required_capabilities, DistributionMatchingCapability)

    def __init__(self, config):
        self.pdd = PDDConfig.from_config(config)
        super().__init__(config)
        if self.gradient_accumulation_iters != 1:
            raise ValueError("Staggered H3 PDD currently uses gradient_accumulation_iters=1")
        if get_sequence_parallel_world_size() != 1:
            raise ValueError("H3 PDD currently uses FSDP without sequence parallelism")
        if self.infer_every_iters:
            raise ValueError("PDD needs its fused-head sampler; generic Euler training previews are not supported")
        self.consumed_microbatches = 0
        self.carry = None
        self.generation_shapes = config["training"]["pdd"].get("generation_shapes")

    def set_model(self, model):
        super().set_model(model)
        self.student = model.capabilities.require(DistributionMatchingCapability)
        if self.model_config["name"] not in {"minimax_h3_t2av", "minimax_h3_ref2av", "minimax_h3_image"}:
            raise ValueError("PDD currently supports H3 Image, T2AV/FL2AV, and Ref2AV")
        if self.model_config["name"] == "minimax_h3_image" and self.model_config.get("image_head_mode") != "base":
            raise ValueError("Image PDD uses image_head_mode=base: the original continuous teacher output projection")

    def _setup_trainable_model(self, model):
        model.capabilities.require(DistributionMatchingCapability).prepare_role("student")
        super()._setup_trainable_model(model)
        model.enable_pdd(self.pdd.grid_size)

    def _restore_trainable_model(self, model):
        super()._restore_trainable_model(model)
        model.restore_pdd_trainable()

    def setup(self, resume_ckpt_path=None):
        super().setup(resume_ckpt_path)
        teacher_config = copy.deepcopy(self.config)
        teacher_config["model"].update(teacher_config["model"].pop("teacher", {}))
        self.teacher_model = build_loaded_model(teacher_config, load_transformer=True, load_vae=False, load_condition_encoder=False)
        self.teacher_model.reuse_frozen_components_from(self.model)
        teacher = self.teacher_model.capabilities.require(DistributionMatchingCapability)
        teacher.prepare_role("teacher")
        teacher.denoiser().requires_grad_(False).eval()
        self.teacher_model.capabilities.require(ParallelCapability).apply(teacher_config)
        self.model.denoiser_module().eval()
        self.grid = self.pdd.base_grid(self.model.device)

    def set_data(self, dataloader_train, dataloader_val=None):
        super().set_data(dataloader_train, dataloader_val)
        if not hasattr(dataloader_train.sampler, "configure_from_microbatch_offset"):
            sampler = TurboCursorSampler(
                dataloader_train.dataset,
                seed=self.config.get("seed", 42),
                shuffle=self.config["data"]["train"].get("shuffle", False),
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

    def _samples(self):
        sampler = self.dataloader_train.sampler
        sampler.configure_from_microbatch_offset(
            start_microbatch=self.consumed_microbatches,
            gradient_accumulation_iters=1,
            fake_update_ratio=1,
            data_consumption_mode="pdd_staggered_trajectory",
            student_data_source="carry_until_terminal",
        )
        epoch = 0
        while True:
            sampler.set_epoch(epoch)
            yield from self.dataloader_train
            epoch += 1

    def _new_carry(self, samples):
        sample = next(samples)
        device = torch.device(self.model.device)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        # Ref conditioning can also add noise. Both streams are independent
        # of prefetch/global RNG, so restoring a carry gives the same next data.
        with torch.random.fork_rng(devices=cuda_devices):

            def seed_stream(name):
                seed = stream_seed(name, self.config.get("seed", 42), get_rank(), self.consumed_microbatches)
                torch.random.default_generator.manual_seed(seed)
                if cuda_devices:
                    torch.cuda.default_generators[cuda_devices[0]].manual_seed(seed)

            seed_stream("pdd-condition")
            condition, _ = self.student.encode_conditions(sample, None, 1.0, lambda value: value)
            shape = self.student.latent_shape(sample, self.generation_shapes, lambda value: value)
            seed_stream("pdd-noise")
            state = self.student.initial_latents(shape, torch.float32, lambda value: value)
        scale_fn = getattr(self.dataloader_train.sampler, "cursor_microbatch_loss_scale", lambda _: 1.0)
        loss_scale = float(scale_fn(self.consumed_microbatches))
        self.consumed_microbatches += 1
        return {"state": state, "condition": condition, "step": 0, "loss_scale": loss_scale}

    @torch.no_grad()
    def _initialize_staggered(self, samples):
        carry = self._new_carry(samples)
        phase = get_data_parallel_rank() % self.pdd.rollout_steps
        selected = map_tensors(carry["state"], torch.clone)
        state = carry["state"]
        # Equal numbers of FSDP forwards on every rank, including ranks whose
        # selected initial phase is zero. Only initialization needs this rollout.
        for step in range(self.pdd.rollout_steps - 1):
            start = step * self.pdd.block_size_min
            pred = self.model.predict_pdd_heads(state, start, carry["condition"], self.grid[start])
            state = advance_blocks(state, pred, start, start + self.pdd.block_size_min, self.grid, sigma_schedule=self.model.pdd_sigma_schedule)
            if step + 1 == phase:
                selected = map_tensors(state, torch.clone)
        torch.clear_autocast_cache()
        return {**carry, "state": selected, "step": phase}

    def update(self, samples, iteration):
        if self.carry is None:
            self.carry = self._new_carry(samples)
        carry = self.carry
        start = carry["step"] * self.pdd.block_size_min
        prediction = self.model.predict_pdd_heads(carry["state"], start, carry["condition"], self.grid[start])
        generator = torch.Generator(device=self.model.device).manual_seed(stream_seed("pdd-head", self.config.get("seed", 42), get_rank(), iteration))
        loss_map, next_state = pdd_loss_from_prediction(
            carry["state"],
            prediction,
            start,
            self.grid,
            lambda state, sigma: self.teacher_model.predict_pdd_teacher(state, sigma, carry["condition"]),
            sigma_schedule=self.model.pdd_sigma_schedule,
            block_size_max=self.pdd.block_size_max,
            num_supervised_heads=self.pdd.num_supervised_heads,
            next_idx=start + self.pdd.block_size_min,
            solver=self.pdd.teacher_solver,
            generator=generator,
            loss_scale=torch.tensor([carry["loss_scale"]], device=self.model.device),
        )
        loss_map["total_loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()
        finite = (torch.isfinite(grad_norm) & torch.isfinite(loss_map["total_loss"])).to(torch.int32)
        if dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError(f"PDD nonfinite loss/gradient at student update {iteration + 1}")
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        next_step = carry["step"] + 1
        self.carry = (
            None
            if next_step == self.pdd.rollout_steps
            else {
                "state": map_tensors(next_state, lambda value: value.detach().clone()),
                "condition": carry["condition"],
                "step": next_step,
                "loss_scale": carry["loss_scale"],
            }
        )
        return {**{key: value.detach() for key, value in loss_map.items()}, "grad_norm": grad_norm.detach()}

    def _recipe(self):
        if hasattr(self, "_recipe_value"):
            return self._recipe_value
        training = {
            key: value
            for key, value in self.training_config.items()
            if key
            not in {
                "output_dir",
                "max_train_iters",
                "save_every_iters",
                "save_total_limit",
                "save_consolidated_weights",
            }
        }
        dataset = self.dataloader_train.dataset
        records = getattr(dataset, "samples", None)
        if records is None:
            raise ValueError("Resumable PDD requires an indexed condition-cache dataset")
        digest = hashlib.sha256()
        for row in records:
            digest.update(json.dumps(row, sort_keys=True, default=str).encode())
            digest.update(b"\n")
        self._recipe_value = json.loads(
            json.dumps(
                {
                    "version": 1,
                    "model": self.model_config,
                    "training": training,
                    "data": {key: value for key, value in self.config["data"]["train"].items() if key not in {"num_workers", "pin_memory"}},
                    "dataset_sha256": digest.hexdigest(),
                    "seed": self.config.get("seed", 42),
                    "distributed": self.config.get("distributed", {}),
                }
            )
        )
        return self._recipe_value

    def _resolve_resume(self):
        if not self.auto_resume:
            return None, 0
        complete = sorted(Path(self.output_train_dir).glob("checkpoint-*/pdd.complete.json"))
        if not complete:
            return None, 0
        marker = complete[-1]
        saved = json.loads(marker.read_text())
        if saved["world_size"] != get_world_size() or saved["recipe"] != self._recipe():
            raise ValueError(f"PDD checkpoint recipe/world size differs: {marker}")
        return str(marker.parent), saved["iteration"]

    def _save_model_weights(self, model, save_dir):
        super()._save_model_weights(model, save_dir)
        model.save_pdd_weights(save_dir)

    def _load_model_weights(self, model, save_dir):
        super()._load_model_weights(model, save_dir)
        model.load_pdd_weights(save_dir)

    def save_checkpoint(self, iteration, save_total_limit):
        # PDD is complete only after weights, optimizer and every rank's carry
        # are durable. Preserve old complete checkpoints until that point.
        super().save_checkpoint(iteration, None)
        directory = Path(self.output_train_dir) / f"checkpoint-{iteration:09d}"
        if self.train_type == "full" and self.parallel.is_fsdp():
            self.model.save_pdd_weights(directory)
        rank_file = directory / f"pdd.rank{get_rank():05d}.pt"
        temporary = rank_file.with_suffix(".tmp")
        torch.save(
            {
                "iteration": iteration,
                "rank": get_rank(),
                "consumed_microbatches": self.consumed_microbatches,
                "carry": map_tensors(self.carry, lambda value: value.detach().cpu()),
            },
            temporary,
        )
        temporary.replace(rank_file)
        barrier()
        if is_main_process():
            metadata = {"iteration": iteration, "world_size": get_world_size(), "recipe": self._recipe()}
            (directory / "pdd_sampling.json").write_text(
                json.dumps(
                    {
                        "grid_sigmas": self.grid.tolist(),
                        "video_shift": self.pdd.video_shift,
                        "audio_shift": self.pdd.audio_shift,
                        "num_inference_steps": self.pdd.num_inference_steps,
                        "block_size_min": self.pdd.block_size_min,
                        "block_size_max": self.pdd.block_size_max,
                        "velocity": "noise_minus_clean",
                        "requires_pdd_heads": True,
                    },
                    indent=2,
                )
                + "\n"
            )
            marker = directory / "pdd.complete.json"
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps(metadata, indent=2) + "\n")
            temporary.replace(marker)
            if save_total_limit is not None:
                complete = sorted(Path(self.output_train_dir).glob("checkpoint-*/pdd.complete.json"))
                for old in complete[:-max(1, int(save_total_limit))]:
                    shutil.rmtree(old.parent)
        barrier()

    def _restore_carry(self, checkpoint, iteration):
        path = Path(checkpoint) / f"pdd.rank{get_rank():05d}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["iteration"] != iteration or saved["rank"] != get_rank():
            raise ValueError(f"Invalid PDD rank state: {path}")
        self.consumed_microbatches = saved["consumed_microbatches"]
        self.carry = map_tensors(saved["carry"], lambda value: value.to(self.model.device))

    def train(self):
        checkpoint, iteration = self._resolve_resume()
        self.setup(checkpoint)
        if checkpoint:
            self._restore_carry(checkpoint, iteration)
        samples = self._samples()
        if checkpoint is None:
            self.carry = self._initialize_staggered(samples)
        logger.info(
            "[pdd] grid={} train_max_nfe={} inference_nfe={} solver={} supervised_heads={} fake=none iter={}/{}",
            self.pdd.grid_size,
            self.pdd.rollout_steps,
            self.pdd.num_inference_steps,
            self.pdd.teacher_solver,
            self.pdd.num_supervised_heads,
            iteration,
            self.max_train_iters,
        )
        last_saved = iteration
        while iteration < self.max_train_iters:
            metrics = self.update(samples, iteration)
            iteration += 1
            if iteration % self.train_log_every_iters == 0:
                values = {key: float(reduce_mean(value)) for key, value in metrics.items()}
                logger.info("[pdd] student_updates={}/{} consumed_conditions={} metrics={}", iteration, self.max_train_iters, self.consumed_microbatches, values)
                self.log_metrics({f"train/{key}": value for key, value in values.items()}, step=iteration)
            if self.save_every_iters and iteration % self.save_every_iters == 0:
                self.save_checkpoint(iteration, self.save_total_limit)
                last_saved = iteration
        if iteration != last_saved:
            self.save_checkpoint(iteration, self.save_total_limit)
        self.finish_monitor()
