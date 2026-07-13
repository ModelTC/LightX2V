import json
import os
import shutil
import time
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers.optimization import get_scheduler
from loguru import logger
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.runtime.checkpoint import find_latest_checkpoint, parse_checkpoint_iteration, prune_checkpoints
from lightx2v_train.runtime.distributed import (
    barrier,
    get_data_parallel_group,
    get_data_parallel_world_size,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    reduce_mean,
)
from lightx2v_train.utils.registry import TRAINER_REGISTER
from lightx2v_train.utils.video import pil_frames_to_video_tensor, save_mp4, video_psnr, video_ssim

ZERO1_PER_RANK_FORMAT = "zero1_per_rank_v1"
FULL_OPTIMIZER_FORMAT = "full_v1"


def _optimizer_shard_filename(rank, world_size):
    return f"optimizer-rank-{rank:05d}-of-{world_size:05d}.pt"


def _temporary_path(path):
    return f"{path}.tmp-{os.getpid()}-{get_rank()}"


def _atomic_torch_save(payload, path):
    temporary_path = _temporary_path(path)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _sync_optimizer_group_options(source_groups, target_groups):
    if len(source_groups) != len(target_groups):
        raise RuntimeError(f"Optimizer parameter-group count mismatch: source={len(source_groups)} target={len(target_groups)}")
    for source, target in zip(source_groups, target_groups):
        for key, value in source.items():
            if key != "params":
                target[key] = value


def _unwrap_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


@TRAINER_REGISTER("fastwam")
class FastWAMTrainer:
    def __init__(self, config):
        self.config = config
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.evaluation_config = config.get("evaluation", config.get("inference", {}))
        self.logging_config = config.get("logging", {})

        optimizer_config = self.training_config.get("optimizer", {})
        self.learning_rate = float(optimizer_config.get("learning_rate", self.training_config.get("learning_rate", 1e-4)))
        self.weight_decay = float(optimizer_config.get("weight_decay", self.training_config.get("weight_decay", 0.0)))
        self.adam_beta1 = float(optimizer_config.get("adam_beta1", 0.9))
        self.adam_beta2 = float(optimizer_config.get("adam_beta2", 0.95))
        self.adam_epsilon = float(optimizer_config.get("adam_epsilon", 1e-8))

        self.output_train_dir = self.training_config["output_dir"]
        self.max_train_iters = self._resolve_max_train_iters()
        self.gradient_accumulation_iters = int(self.training_config.get("gradient_accumulation_iters", 1))
        self.max_grad_norm = float(self.training_config.get("max_grad_norm", 1.0))
        self.save_every_iters = int(self.training_config.get("save_every_iters", 0) or 0)
        self.save_total_limit = int(self.training_config.get("save_total_limit", 3))
        self.save_final = bool(self.training_config.get("save_final", True))
        self.lr_scheduler_name = self.training_config.get("lr_scheduler", "constant")
        self.lr_warmup_iters = int(self.training_config.get("lr_warmup_iters", 0))
        self.train_log_every_iters = max(1, int(self.logging_config.get("train_log_every_iters", 10)))

        zero1_config = self.config.get("distributed", {}).get("zero1", {})
        self.zero1_requested = bool(zero1_config.get("enabled", False)) if isinstance(zero1_config, dict) else bool(zero1_config)
        self.zero1_enabled = False

        self.eval_every_iters = int(self.evaluation_config.get("eval_every_iters", self.evaluation_config.get("infer_every_iters", 0)) or 0)
        self.eval_num_inference_steps = int(self.evaluation_config.get("eval_num_inference_steps", 5))
        self.eval_num_samples = max(1, int(self.evaluation_config.get("num_samples", 1)))
        self.eval_run_inference = bool(self.evaluation_config.get("run_inference", True))
        self.eval_save_video = bool(self.evaluation_config.get("save_video", True))
        self.eval_save_preview = bool(self.evaluation_config.get("save_preview", True))
        self.eval_fps = max(1, int(self.evaluation_config.get("fps", 8)))
        self.eval_tiled = bool(self.evaluation_config.get("tiled", False))
        self.eval_output_dir = self.evaluation_config.get("output_dir", os.path.join(self.output_train_dir, "eval"))
        self.eval_seed = int(self.evaluation_config.get("seed", 42))

        resume_config = self.config.get("resume", {})
        self.auto_resume = bool(resume_config.get("auto_resume", False))
        self.resume_ckpt_path = resume_config.get("resume_ckpt_path")

    def _resolve_max_train_iters(self):
        max_train_iters = self.training_config.get("max_train_iters", self.training_config.get("max_steps"))
        if max_train_iters is not None:
            return int(max_train_iters)
        num_epochs = int(self.training_config.get("num_epochs", 1))
        return max(1, num_epochs)

    def set_model(self, model):
        self.model = model

    def set_data(self, dataloader_train, dataloader_eval=None):
        self.dataloader_train = dataloader_train
        self.dataloader_eval = dataloader_eval

    def _build_optimizer(self):
        optimizer_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": (self.adam_beta1, self.adam_beta2),
            "eps": self.adam_epsilon,
        }
        data_parallel_size = get_data_parallel_world_size()
        if self.zero1_requested and data_parallel_size > 1:
            from torch.distributed.optim import ZeroRedundancyOptimizer

            self.zero1_enabled = True
            logger.info("[optimizer] using ZeroRedundancyOptimizer with AdamW across {} data-parallel ranks", data_parallel_size)
            return ZeroRedundancyOptimizer(
                self.trainable_params,
                optimizer_class=torch.optim.AdamW,
                process_group=get_data_parallel_group(),
                parameters_as_bucket_view=False,
                overlap_with_ddp=False,
                **optimizer_kwargs,
            )

        self.zero1_enabled = False
        if self.zero1_requested:
            logger.info("[optimizer] ZeroRedundancyOptimizer requested but data-parallel size is 1; using AdamW")
        else:
            logger.info("[optimizer] using AdamW without optimizer-state sharding")
        return torch.optim.AdamW(self.trainable_params, **optimizer_kwargs)

    def setup(self, resume_ckpt_path=None):
        self.model.set_dit_only_trainable()
        self.model.log_model_structure()
        self.trainable_params = list(self.model.trainable_parameters())
        if not self.trainable_params:
            raise RuntimeError("FastWAM has no trainable parameters.")

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = get_scheduler(
            self.lr_scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=self.lr_warmup_iters,
            num_training_steps=self.max_train_iters,
        )

        self.train_module = self.model.unwrap_module()
        self.ddp_enabled = False
        if is_distributed():
            self.train_module = DistributedDataParallel(
                self.train_module,
                device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
                find_unused_parameters=bool(self.training_config.get("ddp_find_unused_parameters", False)),
            )
            self.ddp_enabled = True

        if resume_ckpt_path is not None:
            self._load_resume_state(resume_ckpt_path)

    def _resolve_resume(self):
        if self.resume_ckpt_path:
            current_iter = parse_checkpoint_iteration(self.resume_ckpt_path)
            return self.resume_ckpt_path, current_iter
        if not self.auto_resume:
            return None, 0
        ckpt_path, current_iter = find_latest_checkpoint(self.output_train_dir)
        if ckpt_path is None:
            logger.info("Auto-resume enabled but no checkpoint found in '{}'. Starting from scratch.", self.output_train_dir)
            return None, 0
        logger.info("Auto-resuming from checkpoint: {} (iteration {})", ckpt_path, current_iter)
        return ckpt_path, current_iter

    def _load_resume_state(self, resume_ckpt_path):
        weights_path = os.path.join(resume_ckpt_path, "fastwam.pt")
        state_path = os.path.join(resume_ckpt_path, "training_state.pt")
        if not os.path.exists(weights_path):
            raise RuntimeError(f"fastwam.pt not found in {resume_ckpt_path}")
        if not os.path.exists(state_path):
            raise RuntimeError(f"training_state.pt not found in {resume_ckpt_path}")
        self.model.load_checkpoint(weights_path)
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("world_size") != get_world_size():
            raise RuntimeError(f"Cannot resume world_size={state.get('world_size')} checkpoint with world_size={get_world_size()}.")
        optimizer_state_format = state.get("optimizer_state_format")
        expected_optimizer_sharding = "zero1" if self.zero1_enabled else "none"
        if state.get("optimizer_sharding") != expected_optimizer_sharding:
            raise RuntimeError(f"Optimizer sharding mode mismatch: checkpoint={state.get('optimizer_sharding')!r} active={expected_optimizer_sharding!r}.")
        if optimizer_state_format == ZERO1_PER_RANK_FORMAT:
            if not self.zero1_enabled:
                raise RuntimeError("A per-rank ZeRO-1 optimizer checkpoint requires distributed.zero1.enabled=true.")
            shard_files = state.get("optimizer_shard_files")
            if not isinstance(shard_files, list) or len(shard_files) != get_world_size():
                raise RuntimeError(f"Invalid ZeRO-1 optimizer shard manifest: expected {get_world_size()} files, got {shard_files!r}.")
            shard_path = os.path.join(resume_ckpt_path, shard_files[get_rank()])
            if not os.path.isfile(shard_path):
                raise RuntimeError(f"ZeRO-1 optimizer shard not found for rank {get_rank()}: {shard_path}")
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            if shard.get("rank") != get_rank() or shard.get("world_size") != get_world_size():
                raise RuntimeError(
                    f"ZeRO-1 optimizer shard metadata mismatch: rank={shard.get('rank')} world_size={shard.get('world_size')} expected_rank={get_rank()} expected_world_size={get_world_size()}."
                )
            self.optimizer.optim.load_state_dict(shard["optimizer"])
            _sync_optimizer_group_options(self.optimizer.optim.param_groups, self.optimizer.param_groups)
        elif optimizer_state_format == FULL_OPTIMIZER_FORMAT:
            if self.zero1_enabled:
                raise RuntimeError("A full optimizer checkpoint requires distributed.zero1.enabled=false.")
            self.optimizer.load_state_dict(state["optimizer"])
        else:
            raise RuntimeError(f"Unsupported optimizer checkpoint format {optimizer_state_format!r} in {state_path}.")
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        logger.info(
            "[resume] restored FastWAM training state from {} (optimizer sharding: {}, state format: {})",
            resume_ckpt_path,
            expected_optimizer_sharding,
            optimizer_state_format,
        )

    def _save_checkpoint(self, iteration):
        save_start_time = time.perf_counter()
        if is_main_process():
            prune_checkpoints(self.output_train_dir, self.save_total_limit)
        save_dir = os.path.join(self.output_train_dir, f"checkpoint-{iteration:09d}")
        logger.info("[checkpoint] saving FastWAM iter={} path={}", iteration, save_dir)
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
        barrier()

        optimizer_state_format = FULL_OPTIMIZER_FORMAT
        optimizer_shard_files = None
        if self.zero1_enabled:
            optimizer_state_format = ZERO1_PER_RANK_FORMAT
            optimizer_shard_files = [_optimizer_shard_filename(rank, get_world_size()) for rank in range(get_world_size())]
            _sync_optimizer_group_options(self.optimizer.param_groups, self.optimizer.optim.param_groups)
            shard_path = os.path.join(save_dir, optimizer_shard_files[get_rank()])
            shard_start_time = time.perf_counter()
            _atomic_torch_save(
                {
                    "rank": get_rank(),
                    "world_size": get_world_size(),
                    "optimizer": self.optimizer.optim.state_dict(),
                },
                shard_path,
            )
            logger.info(
                "[checkpoint] saved optimizer shard rank={} duration={:.3f}s path={}",
                get_rank(),
                time.perf_counter() - shard_start_time,
                shard_path,
            )
        barrier()

        if is_main_process():
            weights_path = os.path.join(save_dir, "fastwam.pt")
            temporary_weights_path = _temporary_path(weights_path)
            try:
                self.model.save_checkpoint(temporary_weights_path, step=iteration)
                os.replace(temporary_weights_path, weights_path)
            finally:
                if os.path.exists(temporary_weights_path):
                    os.remove(temporary_weights_path)

            training_state = {
                "iteration": iteration,
                "world_size": get_world_size(),
                "optimizer_sharding": "zero1" if self.zero1_enabled else "none",
                "optimizer_state_format": optimizer_state_format,
                "lr_scheduler": self.lr_scheduler.state_dict(),
            }
            if self.zero1_enabled:
                training_state["optimizer_shard_files"] = optimizer_shard_files
            else:
                training_state["optimizer"] = self.optimizer.state_dict()
            _atomic_torch_save(training_state, os.path.join(save_dir, "training_state.pt"))
            config_path = self.config.get("config_path")
            if config_path is not None:
                shutil.copy2(config_path, os.path.join(save_dir, "config.yaml"))
        barrier()
        logger.info(
            "[checkpoint] saved FastWAM iter={} duration={:.3f}s path={}",
            iteration,
            time.perf_counter() - save_start_time,
            save_dir,
        )

    def _forward_loss(self, sample):
        with self.model.autocast_context():
            loss, loss_dict = self.train_module(sample)
        return loss, loss_dict

    def _ddp_sync_context(self, sync_grad):
        if self.ddp_enabled and not sync_grad:
            return self.train_module.no_sync()
        return nullcontext()

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()

        grad_accum_counter = 0
        running_loss = 0.0
        running_loss_dict = {}
        epoch = 0
        start_time = time.perf_counter()

        logger.info(
            "[train] start method=fastwam iter={}/{} world_size={} grad_accum={} log_every={}",
            current_iter,
            self.max_train_iters,
            get_world_size(),
            self.gradient_accumulation_iters,
            self.train_log_every_iters,
        )

        while current_iter < self.max_train_iters:
            sampler = getattr(self.dataloader_train, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for sample in self.dataloader_train:
                sync_grad = (grad_accum_counter + 1) % self.gradient_accumulation_iters == 0
                with self._ddp_sync_context(sync_grad):
                    loss, loss_dict = self._forward_loss(sample)
                    (loss / self.gradient_accumulation_iters).backward()

                running_loss += float(loss.detach().item()) / self.gradient_accumulation_iters
                for key, value in loss_dict.items():
                    running_loss_dict[key] = running_loss_dict.get(key, 0.0) + float(value) / self.gradient_accumulation_iters

                grad_accum_counter += 1
                if grad_accum_counter % self.gradient_accumulation_iters != 0:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                current_iter += 1
                display_loss = reduce_mean(running_loss)
                current_lr = self.lr_scheduler.get_last_lr()[0]
                if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= self.max_train_iters:
                    elapsed = max(time.perf_counter() - start_time, 1e-6)
                    detail = " ".join(f"{key}={reduce_mean(value):.6f}" for key, value in sorted(running_loss_dict.items()))
                    logger.info(
                        "[train] iter={}/{} loss={:.6f} {} lr={:.8f} grad_norm={:.4f} speed={:.3f} it/s",
                        current_iter,
                        self.max_train_iters,
                        display_loss,
                        detail,
                        current_lr,
                        float(grad_norm),
                        current_iter / elapsed,
                    )
                running_loss = 0.0
                running_loss_dict = {}

                if self.eval_every_iters > 0 and self.dataloader_eval is not None and current_iter % self.eval_every_iters == 0:
                    self.evaluate(current_iter)

                if self.save_every_iters > 0 and current_iter % self.save_every_iters == 0:
                    self._save_checkpoint(current_iter)

                if current_iter >= self.max_train_iters:
                    break

            epoch += 1

        if self.save_final and (self.save_every_iters <= 0 or current_iter % self.save_every_iters != 0):
            self._save_checkpoint(current_iter)
        logger.info("[train] finished FastWAM iter={}/{}", current_iter, self.max_train_iters)

    @torch.no_grad()
    def evaluate(self, current_iter):
        if self.dataloader_eval is None:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_start_time = time.perf_counter()
        eval_dataset = self.dataloader_eval.dataset
        model_dataset = _unwrap_dataset(eval_dataset)
        if len(eval_dataset) == 0:
            return

        model = self.model.unwrap_module()
        model.eval()
        step_dir = os.path.join(self.eval_output_dir, f"step_{current_iter:09d}")
        os.makedirs(step_dir, exist_ok=True)

        eval_indices = self._select_eval_indices(len(eval_dataset))
        local_results = []
        for sample_number in range(get_rank(), len(eval_indices), get_world_size()):
            eval_index = eval_indices[sample_number]
            sample = self._to_batched_eval_sample(eval_dataset[eval_index])
            local_results.append(
                self._evaluate_sample(
                    model=model,
                    dataset=model_dataset,
                    sample=sample,
                    sample_number=sample_number,
                    eval_index=eval_index,
                    step_dir=step_dir,
                )
            )

        results = self._gather_eval_results(local_results)
        summary = self._summarize_eval_results(results)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration_seconds = time.perf_counter() - eval_start_time
        if is_main_process():
            metrics_path = os.path.join(step_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "iteration": int(current_iter),
                        "num_samples": len(results),
                        "duration_seconds": duration_seconds,
                        "mean": summary,
                        "samples": results,
                    },
                    handle,
                    ensure_ascii=True,
                    indent=2,
                )
            details = " ".join(f"{key}={value:.6f}" for key, value in sorted(summary.items()))
            logger.info(
                "[eval] iter={} samples={} duration={:.3f}s {} output={}",
                current_iter,
                len(results),
                duration_seconds,
                details,
                step_dir,
            )

        barrier()
        self.model.set_dit_only_trainable()

    def _to_batched_eval_sample(self, sample):
        result = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.unsqueeze(0) if value.ndim >= 1 else value.reshape(1)
            elif isinstance(value, str):
                result[key] = [value]
            else:
                result[key] = value
        return result

    def _select_eval_indices(self, dataset_size):
        count = min(self.eval_num_samples, dataset_size)
        generator = torch.Generator(device="cpu").manual_seed(self.eval_seed)
        return torch.randperm(dataset_size, generator=generator)[:count].tolist()

    def _evaluate_sample(self, model, dataset, sample, sample_number, eval_index, step_dir):
        sample_seed = self.eval_seed + sample_number
        rng_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(sample_seed)
            with self.model.autocast_context():
                val_loss, val_loss_dict = model.training_loss(sample)
        result = {
            "sample_number": int(sample_number),
            "dataset_index": int(eval_index),
            "val_loss": float(val_loss.detach().float().item()),
        }
        result.update({f"val_{key}": float(value) for key, value in val_loss_dict.items()})
        if not self.eval_run_inference:
            return result

        video0 = sample["video"][0]
        action = sample.get("action")
        proprio = sample.get("proprio")
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        prompt = sample.get("prompt", [None])[0]
        input_image = video0[:, 0].unsqueeze(0)
        action_horizon = int(action.shape[1]) if action is not None else int((video0.shape[1] - 1) * dataset.action_video_freq_ratio)
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": int(video0.shape[1]),
            "action_horizon": action_horizon,
            "proprio": None if proprio is None else proprio[0, 0],
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": sample_seed,
            "rand_device": "cpu",
            "tiled": self.eval_tiled,
        }
        if context is not None:
            infer_kwargs.update(prompt=None, context=context[0], context_mask=context_mask[0])
        else:
            infer_kwargs["prompt"] = prompt
        prediction = model.infer(**infer_kwargs)

        pred_video_tensor = pil_frames_to_video_tensor(prediction["video"])
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
        self._validate_video_shapes(pred_video_tensor, gt_video_tensor, "prediction", "ground truth")

        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=self.eval_tiled)
        vae_video_tensor = pil_frames_to_video_tensor(model._decode_latents(vae_latents, tiled=self.eval_tiled))
        self._validate_video_shapes(vae_video_tensor, gt_video_tensor, "VAE reconstruction", "ground truth")

        pred_future = pred_video_tensor[:, 1:]
        vae_future = vae_video_tensor[:, 1:]
        gt_future = gt_video_tensor[:, 1:]
        result.update(
            {
                "future_video_psnr_pred_gt": video_psnr(pred_future, gt_future),
                "future_video_ssim_pred_gt": video_ssim(pred_future, gt_future),
                "future_video_psnr_pred_vae": video_psnr(pred_future, vae_future),
                "future_video_ssim_pred_vae": video_ssim(pred_future, vae_future),
                "future_video_psnr_vae_gt": video_psnr(vae_future, gt_future),
                "future_video_ssim_vae_gt": video_ssim(vae_future, gt_future),
            }
        )

        pred_action = prediction.get("action")
        if action is not None and pred_action is not None:
            result.update(self._action_metrics(dataset, sample, pred_action, action))

        if self.eval_save_video or self.eval_save_preview:
            comparison_frames = self._comparison_frames(pred_video_tensor, vae_video_tensor, gt_video_tensor)
            artifact_stem = f"sample_{sample_number:03d}_index_{eval_index:06d}"
            if self.eval_save_video:
                video_path = os.path.join(step_dir, f"{artifact_stem}.mp4")
                save_mp4(comparison_frames, video_path, fps=self.eval_fps)
                result["video_path"] = video_path
            if self.eval_save_preview:
                preview_path = os.path.join(step_dir, f"{artifact_stem}_preview.png")
                comparison_frames[len(comparison_frames) // 2].save(preview_path)
                result["preview_path"] = preview_path
        return result

    def _action_metrics(self, dataset, sample, pred_action, target_action):
        processor = dataset.processor
        proprio = sample.get("proprio")
        if proprio is None:
            raise ValueError("Eval action denormalization requires sample['proprio'].")

        denormalized = {}
        action_meta = processor.shape_meta["action"]
        state_meta = processor.shape_meta["state"]
        for name, raw_action in (("pred", pred_action), ("gt", target_action)):
            action_btd = raw_action.unsqueeze(0) if raw_action.ndim == 2 else raw_action
            action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)
            batch = {
                "action": action_btd,
                "state": proprio.detach().to(device="cpu", dtype=torch.float32),
            }
            batch = processor.action_state_merger.backward(batch)
            batch = processor.normalizer.backward(batch)
            merged_batch = {
                "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
            }
            denormalized[name] = processor.action_state_merger.forward(merged_batch)["action"].float()

        diff = denormalized["pred"] - denormalized["gt"]
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            valid = ~action_is_pad[0].detach().cpu().bool()
            if valid.any():
                diff = diff[valid]
        return {
            "action_l1": float(diff.abs().mean().item()),
            "action_l2": float(diff.pow(2).mean().item()),
        }

    @staticmethod
    def _validate_video_shapes(left, right, left_name, right_name):
        if left.shape != right.shape:
            raise ValueError(f"Eval {left_name}/{right_name} shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
        if left.shape[1] <= 1:
            raise ValueError(f"Eval video must contain at least one future frame, got {tuple(left.shape)}")

    @staticmethod
    def _comparison_frames(pred, vae, gt):
        labels = ("Prediction", "VAE reconstruction", "Ground truth")
        videos = (pred, vae, gt)
        frame_height = int(gt.shape[2])
        frame_width = int(gt.shape[3])
        label_height = 32
        frames = []
        for frame_index in range(gt.shape[1]):
            canvas = Image.new("RGB", (frame_width * 3, frame_height + label_height), color=(18, 18, 18))
            draw = ImageDraw.Draw(canvas)
            for column, (label, video) in enumerate(zip(labels, videos)):
                array = video[:, frame_index].permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
                canvas.paste(Image.fromarray(np.ascontiguousarray(array)), (column * frame_width, label_height))
                draw.text((column * frame_width + 8, 9), label, fill=(245, 245, 245))
            frames.append(canvas)
        return frames

    @staticmethod
    def _gather_eval_results(local_results):
        if not is_distributed():
            return local_results
        gathered = [None for _ in range(get_world_size())]
        torch.distributed.all_gather_object(gathered, local_results)
        return sorted(
            [result for rank_results in gathered for result in rank_results],
            key=lambda result: result["sample_number"],
        )

    @staticmethod
    def _summarize_eval_results(results):
        if not results:
            return {}
        metric_keys = sorted(key for key, value in results[0].items() if isinstance(value, float))
        return {key: float(sum(result[key] for result in results if key in result) / sum(key in result for result in results)) for key in metric_keys}
