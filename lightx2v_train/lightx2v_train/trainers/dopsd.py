"""Decoupled DOPSD trainer.

Algorithm math lives in :mod:`dopsd_core`; model families only implement the
DopsdCapability data path, and adapter lifecycle is provided independently by
AdapterBankCapability.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

from lightx2v_train.infer import build_inferencer
from lightx2v_train.infer.dopsd_trajectory_viz import save_student_teacher_trajectory_grid
from lightx2v_train.model_capabilities import AdapterBankCapability, DopsdCapability
from lightx2v_train.runtime.checkpoint import prune_checkpoints
from lightx2v_train.runtime.distributed import (
    barrier,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    reduce_mean,
)
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .base import BaseTrainer
from .dopsd_core import DopsdConfig, DopsdObjective


@TRAINER_REGISTER("dopsd")
class DopsdTrainer(BaseTrainer):
    required_capabilities = (
        *BaseTrainer.required_capabilities,
        DopsdCapability,
        AdapterBankCapability,
    )

    def __init__(self, config):
        super().__init__(config)
        self.dopsd_config = DopsdConfig.from_training_config(self.training_config)
        self._train_reference_map = None
        if int(self.gradient_accumulation_iters) <= 0:
            raise ValueError("training.gradient_accumulation_iters must be positive.")
        inference_lora = self.infer_config.get("lora_config") or {}
        if inference_lora.get("path"):
            raise ValueError("DOPSD inference uses the in-memory student and teacher adapters; inference.lora_config.path must not be set.")

    def set_model(self, model):
        super().set_model(model)
        capabilities = model.ensure_capabilities()
        self.dopsd = capabilities.require(DopsdCapability)
        self.adapters = capabilities.require(AdapterBankCapability)
        self.objective = DopsdObjective(
            self.dopsd,
            self.adapters,
            self.noise_scheduler,
            self.dopsd_config,
            self.running_dtype,
        )

    def setup(self, resume_ckpt_path=None):
        algorithm = self.dopsd_config
        self.adapters.configure_pair(
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            student_adapter=algorithm.student_adapter,
            teacher_adapter=algorithm.teacher_adapter,
            initialize_teacher=resume_ckpt_path is None,
        )
        self.parallel.apply(self.config)

        if self.gradient_checkpointing:
            self.trainable_model.enable_gradient_checkpointing()
        if self.infer_every_iters:
            self.inferencer = build_inferencer(self.config)
            self.inferencer.set_model(self.model)

        self.trainable_model.log_structure()
        self.trainable_params = list(self.adapters.parameters(algorithm.student_adapter))
        if not self.trainable_params:
            raise RuntimeError("DOPSD student adapter has no trainable parameters.")
        self.optimizer = self._build_optimizer(self.trainable_params)
        self.lr_scheduler = self._build_lr_scheduler(self.optimizer)

        if resume_ckpt_path is not None:
            self._load_resume_state(resume_ckpt_path)
        self.adapters.set_trainable(algorithm.student_adapter)

    @staticmethod
    def _adapter_weights_path(checkpoint_path, weights_subdir=None):
        weights_dir = Path(checkpoint_path)
        if weights_subdir:
            weights_dir /= weights_subdir
        return weights_dir / "pytorch_lora_weights.safetensors"

    def _load_resume_state(self, resume_ckpt_path):
        if self.parallel.is_fsdp():
            super()._load_distributed_state(resume_ckpt_path)
        else:
            self._load_single_process_state(resume_ckpt_path)

        algorithm = self.dopsd_config
        teacher_path = self._adapter_weights_path(resume_ckpt_path, "teacher")
        if teacher_path.exists():
            self.adapters.load(
                resume_ckpt_path,
                algorithm.teacher_adapter,
                weights_subdir="teacher",
            )
            logger.info("Restored teacher EMA adapter from {}", teacher_path)
        else:
            self.adapters.copy(algorithm.student_adapter, algorithm.teacher_adapter)
            logger.warning(
                "Teacher adapter was not found in {}; initialized it from the student adapter.",
                resume_ckpt_path,
            )

    def _load_single_process_state(self, resume_ckpt_path):
        state_path = Path(resume_ckpt_path) / "training_state.pt"
        if not state_path.exists():
            raise RuntimeError(f"training_state.pt not found in {resume_ckpt_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_metadata(state, str(state_path), resume_ckpt_path)
        self.adapters.load(resume_ckpt_path, self.dopsd_config.student_adapter)
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        logger.info("Restored DOPSD training state from {}", state_path)

    def _checkpoint_metadata(self):
        metadata = self.dopsd_config.checkpoint_metadata()
        target_modules = self.lora_target_modules
        if isinstance(target_modules, (list, tuple)):
            target_modules = list(target_modules)
        metadata["lora"] = {
            "rank": int(self.lora_rank),
            "alpha": int(self.lora_alpha),
            "target_modules": target_modules,
        }
        return metadata

    def _validate_checkpoint_metadata(self, state, state_path, resume_ckpt_path):
        super()._validate_checkpoint_metadata(state, state_path, resume_ckpt_path)
        saved_metadata = state.get("dopsd")
        if saved_metadata is None:
            logger.warning("DOPSD checkpoint {} has no algorithm metadata; treating it as a legacy checkpoint.", state_path)
            return
        expected_metadata = self._checkpoint_metadata()
        if saved_metadata != expected_metadata:
            raise RuntimeError(f"DOPSD checkpoint configuration does not match the current run: saved={saved_metadata}, current={expected_metadata}.")

    def _build_train_reference_map(self):
        if self._train_reference_map is not None:
            return
        self._train_reference_map = {}
        for record in getattr(self.dataloader_train.dataset, "samples", ()):
            target_image = record.get("target_image")
            prompt = record.get("prompt")
            if target_image is not None and prompt is not None:
                self._train_reference_map[str(prompt).strip()] = target_image

    def _resolve_teacher_reference_image(self, record, dataset):
        target_image = record.get("target_image")
        if target_image is not None:
            if isinstance(target_image, (str, os.PathLike)):
                return dataset.load_image(target_image)
            return target_image

        self._build_train_reference_map()
        fallback_path = self._train_reference_map.get(str(record.get("prompt", "")).strip())
        return None if fallback_path is None else dataset.load_image(fallback_path)

    def compute_loss_on_sample(self, sample, collect_trajectory=False):
        return self.objective.compute_loss(sample, collect_trajectory=collect_trajectory)

    @torch.no_grad()
    def _save_training_trajectory(self, current_iter, result):
        if result.state_ids is None:
            raise RuntimeError("DOPSD trajectory result is missing state IDs.")
        trajectory_dir = Path(self.output_train_dir) / "trajectory" / f"iter-{current_iter:09d}"
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        student_images = []
        teacher_images = []
        num_steps = len(result.student_trajectory)
        for step_index, (student_x0, teacher_x0) in enumerate(zip(result.student_trajectory, result.teacher_trajectory)):
            logger.info(
                "[train] trajectory decode iter={} step={}/{}",
                current_iter,
                step_index + 1,
                num_steps,
            )
            student_images.extend(self.dopsd.decode_state(student_x0, result.state_ids))
            teacher_images.extend(self.dopsd.decode_state(teacher_x0, result.state_ids))

        save_path = trajectory_dir / "student_teacher_x0_traj.png"
        save_student_teacher_trajectory_grid(student_images, teacher_images, save_path)
        logger.info("[train] saved trajectory iter={} path={}", current_iter, save_path)

    def train(self):
        resume_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        algorithm = self.dopsd_config
        grad_accum_counter = 0
        running_loss = 0.0
        logger.info(
            "[train] dopsd start iter={}/{} world_size={} grad_accum={} steps={} "
            "ema={} warmup_ema={} warmup_iters={} weights={} teacher_prompt={!r} "
            "teacher_use_dataset_prompt={} trajectory_every_iters={}",
            current_iter,
            self.max_train_iters,
            get_world_size(),
            self.gradient_accumulation_iters,
            algorithm.num_training_steps,
            algorithm.ema_decay,
            algorithm.ema_decay_warmup,
            algorithm.ema_decay_warmup_iters,
            algorithm.loss_weights(algorithm.num_training_steps),
            algorithm.edit_sys_prompt,
            algorithm.teacher_use_dataset_prompt,
            algorithm.trajectory_every_iters,
        )

        if self.infer_every_iters:
            self.inferencer.set_data(self.dataloader_eval)
            if current_iter == 0:
                self.run_inference(current_iter)

        epoch = 0
        while current_iter < self.max_train_iters:
            sampler = getattr(self.dataloader_train, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for sample in self.dataloader_train:
                sync_grad = (grad_accum_counter + 1) % self.gradient_accumulation_iters == 0
                self._set_gradient_sync(sync_grad)
                collect_trajectory = bool(sync_grad and algorithm.trajectory_every_iters and (current_iter + 1) % algorithm.trajectory_every_iters == 0)
                result = self.compute_loss_on_sample(sample, collect_trajectory=collect_trajectory)
                (result.loss / self.gradient_accumulation_iters).backward()
                running_loss += result.loss.item() / self.gradient_accumulation_iters

                grad_accum_counter += 1
                if not sync_grad:
                    continue

                torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                ema_decay = algorithm.ema_decay_at(current_iter + 1)
                self.adapters.ema_update(
                    algorithm.student_adapter,
                    algorithm.teacher_adapter,
                    ema_decay,
                )
                current_iter += 1

                display_loss = reduce_mean(running_loss)
                current_lr = self.lr_scheduler.get_last_lr()[0]
                if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= self.max_train_iters:
                    logger.info(
                        "[train] iter={}/{} loss_dopsd={:.6f} lr={:.8f} ema_decay={:.4f}",
                        current_iter,
                        self.max_train_iters,
                        display_loss,
                        current_lr,
                        ema_decay,
                    )
                    self.log_metrics(
                        {
                            "train/loss_dopsd": display_loss,
                            "train/lr": current_lr,
                            "train/ema_decay": ema_decay,
                        },
                        step=current_iter,
                    )
                running_loss = 0.0

                if collect_trajectory:
                    self.trainable_model.set_eval()
                    barrier()
                    if is_main_process():
                        self._save_training_trajectory(current_iter, result)
                    barrier()
                    self.adapters.set_trainable(algorithm.student_adapter)

                if self.save_every_iters and current_iter % self.save_every_iters == 0:
                    self.save_checkpoint(current_iter, self.save_total_limit)
                if self.infer_every_iters and current_iter % self.infer_every_iters == 0:
                    self.run_inference(current_iter)
                if current_iter >= self.max_train_iters:
                    break
            epoch += 1

        self.finish_monitor()
        logger.info("[train] finished iter={}/{}", current_iter, self.max_train_iters)

    def _first_teacher_reference(self, dataset, samples):
        for record in samples:
            reference = self._resolve_teacher_reference_image(record, dataset)
            if reference is not None:
                return reference, record
        return None, None

    @torch.no_grad()
    def _run_teacher_inference(self, current_iter, iter_output_dir):
        dataset = self.inferencer.dataloader_eval.dataset
        samples = dataset.samples
        rank = get_rank()
        world_size = get_world_size()
        num_steps = int(self.infer_config.get("num_inference_steps", self.dopsd_config.num_training_steps))
        if num_steps <= 0:
            raise ValueError("inference.num_inference_steps must be positive.")
        base_seed = int(self.infer_config.get("seed", 42))

        teacher_output_dir = Path(iter_output_dir) / "teacher"
        teacher_output_dir.mkdir(parents=True, exist_ok=True)
        dummy_reference, dummy_record = self._first_teacher_reference(dataset, samples)
        if dummy_reference is None:
            logger.warning("[train] skipped teacher inference: evaluation data has no reference images.")
            barrier()
            return

        self.trainable_model.set_eval()
        num_slots = (len(samples) + world_size - 1) // world_size if is_distributed() else len(samples)
        saved_count = 0
        skipped_count = 0
        logger.info(
            "[train] running teacher inference iter={} output_dir={} steps={}",
            current_iter,
            teacher_output_dir,
            num_steps,
        )

        for slot in range(num_slots):
            sample_index = slot * world_size + rank if is_distributed() else slot
            has_sample = sample_index < len(samples)
            record = samples[sample_index] if has_sample else dummy_record
            reference = self._resolve_teacher_reference_image(record, dataset) if has_sample else dummy_reference
            should_save = has_sample and reference is not None
            if reference is None:
                reference = dummy_reference
            if has_sample and not should_save:
                skipped_count += 1
                logger.warning(
                    "[train] teacher infer skip sample={}/{}: missing target_image in evaluation metadata",
                    sample_index + 1,
                    len(samples),
                )

            if not torch.is_tensor(reference):
                raise TypeError(f"DOPSD teacher references must be tensors after dataset loading, got {type(reference).__name__}.")
            reference = reference.unsqueeze(0) if reference.ndim == 3 else reference
            prompt = record.get("prompt", "")
            teacher_prompts = self.dopsd_config.teacher_prompts(prompt)
            seed = base_seed + sample_index if has_sample else base_seed
            generator = torch.Generator(device=self.dopsd.device).manual_seed(seed)
            rollout = self.objective.rollout_teacher(
                reference,
                teacher_prompts,
                num_steps,
                generator=generator,
            )
            if not should_save:
                continue

            logger.info(
                "[train] teacher infer sample={}/{} seed={} size={}x{} prompt={!r}",
                sample_index + 1,
                len(samples),
                seed,
                rollout.height,
                rollout.width,
                teacher_prompts[0],
            )
            images = self.dopsd.decode_state(rollout.state, rollout.state_ids)
            save_path = teacher_output_dir / f"{sample_index:05d}.png"
            images[0].save(save_path)
            saved_count += 1
            logger.info("[train] teacher infer sample={}/{} saved path={}", sample_index + 1, len(samples), save_path)

        barrier()
        if is_distributed():
            counts = torch.tensor(
                [saved_count, skipped_count],
                device=self.dopsd.device,
                dtype=torch.int64,
            )
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            saved_count, skipped_count = counts.tolist()
        logger.info(
            "[train] finished teacher inference iter={} saved={} skipped={}",
            current_iter,
            saved_count,
            skipped_count,
        )

    def run_inference(self, current_iter):
        output_dir = Path(self.infer_config.get("output_dir", "./output_infer")) / f"iter-{current_iter:09d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.inferencer.output_infer_dir = str(output_dir)
        logger.info("[train] running student inference iter={} output_dir={}", current_iter, output_dir)
        with self.adapters.activate(self.dopsd_config.student_adapter):
            self.inferencer.infer()
        self._run_teacher_inference(current_iter, output_dir)
        self.adapters.set_trainable(self.dopsd_config.student_adapter)
        logger.info("[train] finished inference iter={}", current_iter)

    def _training_state(self, iteration, include_optimizer):
        state = {
            "iteration": iteration,
            "world_size": get_world_size(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "dopsd": self._checkpoint_metadata(),
        }
        if include_optimizer:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    @staticmethod
    def _atomic_torch_save(state, path):
        path = Path(path)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)

    def save_checkpoint(self, iteration, save_total_limit):
        if is_main_process():
            prune_checkpoints(self.output_train_dir, save_total_limit)
        save_dir = Path(self.output_train_dir) / f"checkpoint-{iteration:09d}"
        logger.info("[train] saving DOPSD checkpoint iter={} path={}", iteration, save_dir)
        if is_main_process():
            save_dir.mkdir(parents=True, exist_ok=True)
        barrier()

        self.adapters.save(save_dir, self.dopsd_config.student_adapter)
        barrier()
        self.adapters.save(save_dir, self.dopsd_config.teacher_adapter, weights_subdir="teacher")
        barrier()

        config_path = self.config.get("config_path")
        if is_main_process() and config_path is not None:
            shutil.copy2(config_path, save_dir / "config.yaml")

        if self.parallel.is_fsdp():
            self._save_distributed_state(save_dir, iteration)
        else:
            if is_main_process():
                self._atomic_torch_save(
                    self._training_state(iteration, include_optimizer=True),
                    save_dir / "training_state.pt",
                )
            barrier()

        if is_main_process():
            (save_dir / "_SUCCESS").touch()
        barrier()
        logger.info("[train] saved DOPSD checkpoint iter={} path={}", iteration, save_dir)

    def _save_distributed_state(self, save_dir, iteration):
        dist_state_path = Path(save_dir) / "dist_state"
        if is_main_process():
            dist_state_path.mkdir(parents=True, exist_ok=True)
        barrier()

        options = StateDictOptions(ignore_frozen_params=True, strict=False)
        model_state, optimizer_state = get_state_dict(
            self.parallel.state_module(),
            self.optimizer,
            options=options,
        )
        dcp.save(
            {"model": model_state, "optimizer": optimizer_state},
            checkpoint_id=str(dist_state_path),
        )
        barrier()

        # trainer_state.pt is a runtime completion marker, so write it last.
        if is_main_process():
            self._atomic_torch_save(
                self._training_state(iteration, include_optimizer=False),
                Path(save_dir) / "trainer_state.pt",
            )
        barrier()
