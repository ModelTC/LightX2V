"""PyTorch fine-tuning loop for the localized OpenPI pi0.5-LIBERO model."""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_model, save_file, save_model
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.runtime.distributed import (
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    reduce_mean,
)
from lightx2v_train.runtime.monitor import build_monitor
from lightx2v_train.utils.registry import TRAINER_REGISTER

CHECKPOINT_SCHEMA_VERSION = 4
CHECKPOINT_CREATOR = "lightx2v-openpi-trainer"
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d{9})$")
REQUIRED_CHECKPOINT_FILES = (
    "model.safetensors",
    "config.json",
    "assets/paligemma_tokenizer.model",
    "assets/physical-intelligence/libero/norm_stats.json",
    "ema/model.safetensors",
    "ema/config.json",
    "ema/assets/paligemma_tokenizer.model",
    "ema/assets/physical-intelligence/libero/norm_stats.json",
    "training_state.pt",
    "manifest.json",
    "_SUCCESS",
)
NUMERICAL_POLICY = {
    "parameter_dtype": "float32",
    "gradient_dtype": "float32",
    "optimizer_state_dtype": "float32",
    "ema_dtype": "float32",
    "loss_dtype": "float32",
    "grad_scaler": False,
    "tf32": False,
}


def _assert_fp32_gradients(model: torch.nn.Module) -> None:
    wrong = {
        name: str(parameter.grad.dtype) for name, parameter in model.named_parameters() if parameter.grad is not None and parameter.grad.is_floating_point() and parameter.grad.dtype != torch.float32
    }
    if wrong:
        preview = dict(list(wrong.items())[:20])
        raise RuntimeError(f"OpenPI gradients must accumulate in float32; found {len(wrong)} mismatches: {preview}")


def _assert_fp32_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    wrong: dict[str, str] = {}
    for parameter_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point() and value.dtype != torch.float32:
                wrong[f"parameter_{parameter_index}.{name}"] = str(value.dtype)
    if wrong:
        preview = dict(list(wrong.items())[:20])
        raise RuntimeError(f"OpenPI optimizer floating-point state must be float32; found {len(wrong)} mismatches: {preview}")


def _validate_full_checkpoint_weight_precision(checkpoint_dir: Path) -> None:
    """Reject precision-lossy model masters before mutating resume state."""
    weight_files = {
        "online model": checkpoint_dir / "model.safetensors",
        "EMA model": checkpoint_dir / "ema/model.safetensors",
    }
    for kind, path in weight_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"OpenPI schema {CHECKPOINT_SCHEMA_VERSION} {kind} checkpoint is missing: {path}")
        wrong: dict[str, str] = {}
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                dtype = checkpoint.get_slice(name).get_dtype()
                if (dtype.startswith("F") or dtype.startswith("BF")) and dtype != "F32":
                    wrong[name] = dtype
                    if len(wrong) == 20:
                        break
        if wrong:
            raise RuntimeError(f"OpenPI schema {CHECKPOINT_SCHEMA_VERSION} {kind} checkpoint must keep every floating-point tensor in FP32; found lower-precision tensors in {path}: {wrong}")


def _shared_aliases(state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
    """Return alias-to-keeper names for exact tied tensors in a live model."""
    storage_groups: dict[tuple[torch.device, int, int], list[str]] = {}
    for name, tensor in state_dict.items():
        if tensor.device.type == "meta" or tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        identity = (tensor.device, storage.data_ptr(), storage.nbytes())
        storage_groups.setdefault(identity, []).append(name)

    aliases = {}
    for names in storage_groups.values():
        if len(names) < 2:
            continue
        names.sort()
        keeper = names[0]
        keeper_tensor = state_dict[keeper]
        keeper_storage = keeper_tensor.untyped_storage()
        if keeper_tensor.data_ptr() != keeper_storage.data_ptr() or keeper_tensor.numel() * keeper_tensor.element_size() != keeper_storage.nbytes():
            raise RuntimeError(f"EMA tied tensor keeper {keeper!r} does not cover its complete storage")
        keeper_signature = (
            keeper_tensor.dtype,
            keeper_tensor.shape,
            keeper_tensor.stride(),
            keeper_tensor.storage_offset(),
        )
        for name in names[1:]:
            tensor = state_dict[name]
            signature = (tensor.dtype, tensor.shape, tensor.stride(), tensor.storage_offset())
            if signature != keeper_signature:
                raise RuntimeError(f"EMA checkpointing only supports exact tied tensor aliases; {keeper!r} and {name!r} share storage but have different views")
            aliases[name] = keeper
    return aliases


def _tensor_storage_bytes(value: Any) -> int:
    """Count unique tensor storage bytes in a nested checkpoint payload."""
    seen: set[tuple[torch.device, int, int]] = set()
    total = 0

    def visit(item: Any) -> None:
        nonlocal total
        if torch.is_tensor(item):
            if item.device.type == "meta" or item.numel() == 0:
                return
            storage = item.untyped_storage()
            identity = (item.device, storage.data_ptr(), storage.nbytes())
            if identity not in seen:
                seen.add(identity)
                total += storage.nbytes()
        elif isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return total


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _set_seed(seed: int) -> None:
    seed += get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": list(random.getstate()),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def _gather_rng_states() -> list[dict[str, Any]]:
    local_state = _capture_rng_state()
    if not is_distributed():
        return [local_state]
    states: list[dict[str, Any] | None] = [None] * get_world_size()
    dist.all_gather_object(states, local_state)
    if any(state is None for state in states):
        raise RuntimeError("Failed to collect RNG state from every distributed rank")
    return states  # type: ignore[return-value]


def _nested_tuple(value):
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(_nested_tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        if state["torch_cuda"] is None:
            raise RuntimeError("Checkpoint has no CUDA RNG state for a CUDA resume")
        torch.cuda.set_rng_state(state["torch_cuda"])


def _build_scheduler(optimizer: torch.optim.Optimizer, spec: dict[str, float | int]):
    warmup_steps = int(spec["warmup_steps"])
    peak_lr = float(spec["peak_lr"])
    decay_steps = int(spec["decay_steps"])
    decay_lr = float(spec["decay_lr"])
    if peak_lr <= 0 or decay_lr < 0:
        raise ValueError("Learning rates must be non-negative and peak_lr must be positive")
    if warmup_steps < 0 or decay_steps < 1:
        raise ValueError("warmup_steps must be non-negative and decay_steps must be positive")

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            initial_lr = peak_lr / (warmup_steps + 1)
            value = initial_lr + (peak_lr - initial_lr) * step / warmup_steps
        else:
            progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
            value = decay_lr + (peak_lr - decay_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        return value / peak_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be between zero and one, got {decay}")
        self.decay = decay
        self.num_updates = 0
        self.shadow = {name: value.detach().to(dtype=torch.float32).clone() if value.is_floating_point() else value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        for name, value in current.items():
            target = self.shadow[name]
            if target.is_floating_point():
                target.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                target.copy_(value)
        self.num_updates += 1

    def inference_state(self, model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
        model_state = model.state_dict()
        if model_state.keys() != self.shadow.keys():
            raise RuntimeError("Model state changed after EMA initialization")
        aliases = _shared_aliases(model_state)
        state = {name: value.detach().cpu().contiguous() for name, value in self.shadow.items() if name not in aliases}
        return state, aliases

    def save(self, path: Path, model: torch.nn.Module) -> None:
        state, aliases = self.inference_state(model)
        metadata = {"format": "pt", "kind": "ema", "master_dtype": "float32", **aliases}
        save_file(state, str(path), metadata=metadata)

    def load(self, path: Path, model: torch.nn.Module, *, num_updates: int) -> None:
        expected = model.state_dict()
        aliases = _shared_aliases(expected)
        required = set(expected) - set(aliases)
        if self.shadow.keys() != expected.keys():
            raise RuntimeError("Model state changed after EMA initialization")

        # Keep the existing FP32 shadow allocations and stream one tensor at a
        # time from the CPU-mapped SafeTensors file. Loading the complete EMA on
        # GPU and then cloning it would add roughly two model copies to resume's
        # peak memory.
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            loaded = set(checkpoint.keys())
            missing = required - loaded
            unexpected = loaded - set(expected)
            wrong_shape = {
                name: (tuple(checkpoint.get_slice(name).get_shape()), tuple(expected[name].shape))
                for name in required & loaded
                if tuple(checkpoint.get_slice(name).get_shape()) != tuple(expected[name].shape)
            }
            if missing or unexpected or wrong_shape:
                raise RuntimeError(f"EMA checkpoint does not match the model: missing={sorted(missing)}, unexpected={sorted(unexpected)}, wrong_shape={wrong_shape}")
            for name in sorted(required):
                self.shadow[name].copy_(checkpoint.get_tensor(name))

        for alias, keeper in aliases.items():
            self.shadow[alias].copy_(self.shadow[keeper])
        self.num_updates = int(num_updates)


def _is_complete_checkpoint(path: Path) -> bool:
    if not path.is_dir() or not CHECKPOINT_PATTERN.fullmatch(path.name):
        return False
    if any(not (path / name).is_file() for name in REQUIRED_CHECKPOINT_FILES):
        return False
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("created_by") == CHECKPOINT_CREATOR and manifest.get("schema_version") == CHECKPOINT_SCHEMA_VERSION


def _prune_checkpoints(output_dir: Path, total_limit: int | None, keep_period: int | None) -> None:
    if total_limit is None or total_limit < 1 or not output_dir.is_dir():
        return
    checkpoints = sorted(
        (path for path in output_dir.iterdir() if _is_complete_checkpoint(path)),
        key=lambda path: int(CHECKPOINT_PATTERN.fullmatch(path.name).group(1)),
    )
    disposable = []
    for path in checkpoints:
        step = int(CHECKPOINT_PATTERN.fullmatch(path.name).group(1))
        if keep_period is None or step % keep_period:
            disposable.append(path)
    for path in disposable[: max(0, len(disposable) - total_limit)]:
        shutil.rmtree(path)


@TRAINER_REGISTER("openpi_flow_matching")
class OpenPIFlowMatchingTrainer:
    """Fine-tune PI0Pytorch with strict full-state resume and EMA."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.training_config = config["training"]
        self.resume_config = config["resume"]
        self.output_dir = Path(self.training_config["output_dir"]).expanduser().resolve()
        self.max_train_iters = int(self.training_config["max_train_iters"])
        self.gradient_accumulation_iters = int(self.training_config["gradient_accumulation_iters"])
        self.max_grad_norm = float(self.training_config["max_grad_norm"])
        self.save_every_iters = int(self.training_config["save_every_iters"])
        self.save_total_limit = self.training_config["save_total_limit"]
        keep_period = self.training_config["keep_period"]
        self.keep_period = None if keep_period is None else int(keep_period)
        if self.keep_period is not None and self.keep_period < 1:
            raise ValueError("training.keep_period must be positive or null")
        self.log_every_iters = int(config["logging"]["train_log_every_iters"])
        if self.log_every_iters < 1:
            raise ValueError("logging.train_log_every_iters must be positive")
        optimizer_config = self.training_config["optimizer"]
        self.optimizer_spec = {
            "learning_rate": float(optimizer_config["learning_rate"]),
            "adam_beta1": float(optimizer_config["adam_beta1"]),
            "adam_beta2": float(optimizer_config["adam_beta2"]),
            "weight_decay": float(optimizer_config["weight_decay"]),
            "adam_epsilon": float(optimizer_config["adam_epsilon"]),
        }
        schedule_config = self.training_config["lr_schedule"]
        self.schedule_spec = {
            "warmup_steps": int(schedule_config["warmup_steps"]),
            "peak_lr": float(schedule_config["peak_lr"]),
            "decay_steps": int(schedule_config["decay_steps"]),
            "decay_lr": float(schedule_config["decay_lr"]),
        }
        ema_config = self.training_config["ema"]
        self.ema_decay = float(ema_config["decay"])
        if self.gradient_accumulation_iters < 1:
            raise ValueError("training.gradient_accumulation_iters must be positive")
        data_accumulation = int(config["data"]["train"]["gradient_accumulation_iters"])
        if data_accumulation != self.gradient_accumulation_iters:
            raise ValueError("data.train.gradient_accumulation_iters must equal training.gradient_accumulation_iters")
        self.monitor = build_monitor(config)

    def set_model(self, model) -> None:
        self.model = model
        self.core_model = model.require_core_model()

    def set_data(self, dataloader_train, dataloader_eval=None) -> None:
        if dataloader_eval is not None:
            raise ValueError("OpenPI trainer does not run an in-training LIBERO rollout")
        if not hasattr(dataloader_train, "metadata"):
            raise TypeError("OpenPI trainer requires the openpi_libero data loader")
        self.dataloader = dataloader_train

    def _wrap_ddp(self) -> torch.nn.Module:
        if not is_distributed():
            return self.core_model
        dp_config = self.config["distributed"]["dp"]
        kwargs = {
            "broadcast_buffers": bool(dp_config["broadcast_buffers"]),
            "find_unused_parameters": bool(dp_config["find_unused_parameters"]),
            "gradient_as_bucket_view": bool(dp_config["gradient_as_bucket_view"]),
            "static_graph": bool(dp_config["static_graph"]),
        }
        if torch.cuda.is_available():
            kwargs.update(device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device())
        logger.info("[openpi:ddp] wrapping model with {}", kwargs)
        return DistributedDataParallel(self.core_model, **kwargs)

    def _resolve_resume_path(self) -> Path | None:
        explicit = self.resume_config["checkpoint_path"]
        if explicit:
            path = Path(explicit).expanduser().resolve()
            if not _is_complete_checkpoint(path):
                raise RuntimeError(f"Not a complete LightX2V OpenPI checkpoint: {path}")
            return path
        candidates = []
        if self.output_dir.is_dir():
            candidates = [path for path in self.output_dir.iterdir() if CHECKPOINT_PATTERN.fullmatch(path.name)]
        incomplete = [path for path in candidates if not _is_complete_checkpoint(path)]
        if incomplete:
            raise RuntimeError(f"Output directory contains incomplete or foreign checkpoints: {incomplete}")
        if candidates:
            raise RuntimeError(f"Resume is disabled but {self.output_dir} already contains checkpoints; use run_pi05_resume_ema.sh or choose a new OPENPI_TRAIN_OUTPUT")
        return None

    def _checkpoint_contract(self) -> dict[str, Any]:
        data = self.dataloader.metadata
        image_pipeline = data["image_pipeline"]
        architecture = self.model.architecture_metadata()
        compute_dtype = architecture["compute_dtype"]
        parameter_dtype = architecture["parameter_dtype"]
        if compute_dtype != "bfloat16" or parameter_dtype != "float32":
            raise RuntimeError(
                f"The OpenPI training contract requires float32 canonical parameters and bfloat16 transformer compute; got parameter_dtype={parameter_dtype!r}, compute_dtype={compute_dtype!r}"
            )
        return {
            "architecture": architecture,
            "numerical_policy": {**NUMERICAL_POLICY, "compute_dtype": compute_dtype},
            "optimizer": self.optimizer_spec,
            "lr_schedule": self.schedule_spec,
            "ema_decay": self.ema_decay,
            "ema_master_dtype": "float32",
            "gradient_accumulation_iters": self.gradient_accumulation_iters,
            "gradient_checkpointing": bool(self.training_config["gradient_checkpointing"]),
            "max_grad_norm": self.max_grad_norm,
            "dataset": {
                key: data[key]
                for key in (
                    "repo_id",
                    "episodes",
                    "frames",
                    "language_tasks",
                    "fps",
                    "global_batch_size",
                    "optimizer_batches_per_epoch",
                    "micro_batches_per_epoch",
                    "shuffle",
                    "seed",
                    "dataset_info_sha256",
                    "norm_stats_sha256",
                    "tokenizer_sha256",
                )
            },
            "image_pipeline": image_pipeline,
        }

    def _load_full_checkpoint(self, path: Path) -> tuple[int, int, int]:
        state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
        if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported training-state schema in {path}")
        if state.get("contract") != self._checkpoint_contract():
            raise RuntimeError(f"Checkpoint training contract does not match the current config: {path}")
        checkpoint_world_size = int(state["world_size"])
        if checkpoint_world_size != get_world_size():
            raise RuntimeError(f"Checkpoint world_size={checkpoint_world_size}, current world_size={get_world_size()}; strict resume requires the same world size")

        _validate_full_checkpoint_weight_precision(path)
        load_model(self.core_model, str(path / "model.safetensors"), strict=True, device=str(self.model.device))
        self.core_model.assert_fp32_parameters()
        self.optimizer.load_state_dict(state["optimizer"])
        _assert_fp32_optimizer_state(self.optimizer)
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.ema.load(path / "ema/model.safetensors", self.core_model, num_updates=int(state["ema_num_updates"]))
        global_step = int(state["global_step"])
        if self.lr_scheduler.last_epoch != global_step:
            raise RuntimeError(f"Checkpoint scheduler step {self.lr_scheduler.last_epoch} does not match global_step={global_step}")
        rng_states = state.get("rng_states")
        if not isinstance(rng_states, list) or len(rng_states) != checkpoint_world_size:
            raise RuntimeError(f"Checkpoint does not contain one RNG state per rank: {path}")
        self._resume_rng_state = rng_states[get_rank()]
        logger.info("[openpi:resume] full state restored from {} at step={}", path, global_step)
        return global_step, int(state["data_epoch"]), int(state["batches_in_epoch"])

    def _write_inference_artifact(self, destination: Path) -> None:
        tokenizer = Path(self.dataloader.metadata["tokenizer_path"])
        norm_stats = Path(self.dataloader.metadata["norm_stats_path"])
        tokenizer_target = destination / "assets/paligemma_tokenizer.model"
        norm_target = destination / "assets/physical-intelligence/libero/norm_stats.json"
        tokenizer_target.parent.mkdir(parents=True, exist_ok=True)
        norm_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tokenizer, tokenizer_target)
        shutil.copy2(norm_stats, norm_target)
        artifact_config = {
            **self.model.architecture_metadata(),
            "output_action_dim": 7,
            "state_dim": 8,
            "num_inference_steps": 10,
            "pytorch_compile_mode": None,
        }
        _write_json(destination / "config.json", artifact_config)

    def _check_checkpoint_disk_space(self) -> None:
        model_state = self.core_model.state_dict()
        model_bytes = _tensor_storage_bytes(model_state)
        aliases = _shared_aliases(model_state)
        ema_state = {name: value for name, value in self.ema.shadow.items() if name not in aliases}
        ema_bytes = _tensor_storage_bytes(ema_state)
        optimizer_bytes = _tensor_storage_bytes(self.optimizer.state_dict())
        payload_bytes = model_bytes + ema_bytes + optimizer_bytes
        required_bytes = int(payload_bytes * 1.15) + 512 * 1024**2
        free_bytes = shutil.disk_usage(self.output_dir).free
        logger.info(
            "[openpi:checkpoint] disk preflight payload={:.2f} GiB required_with_margin={:.2f} GiB free={:.2f} GiB",
            payload_bytes / 1024**3,
            required_bytes / 1024**3,
            free_bytes / 1024**3,
        )
        if free_bytes < required_bytes:
            raise OSError(f"Not enough free space for an atomic OpenPI checkpoint: need at least {required_bytes / 1024**3:.2f} GiB, have {free_bytes / 1024**3:.2f} GiB in {self.output_dir}")

    def _save_checkpoint(self, global_step: int, data_epoch: int, batches_in_epoch: int) -> None:
        rng_states = _gather_rng_states()
        save_error: Exception | None = None
        save_error_message: str | None = None
        if is_main_process():
            stage: Path | None = None
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self._check_checkpoint_disk_space()
                name = f"checkpoint-{global_step:09d}"
                final = self.output_dir / name
                if final.exists():
                    raise FileExistsError(f"Refusing to replace an existing checkpoint: {final}")
                stage = self.output_dir / f".{name}.tmp-{uuid.uuid4().hex}"
                stage.mkdir()
                save_model(self.core_model, str(stage / "model.safetensors"), metadata={"format": "pt"})
                ema_dir = stage / "ema"
                ema_dir.mkdir()
                self.ema.save(ema_dir / "model.safetensors", self.core_model)
                contract = self._checkpoint_contract()
                training_state = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "global_step": global_step,
                    "world_size": get_world_size(),
                    "data_epoch": data_epoch,
                    "batches_in_epoch": batches_in_epoch,
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "ema_num_updates": self.ema.num_updates,
                    "rng_states": rng_states,
                    "contract": contract,
                }
                torch.save(training_state, stage / "training_state.pt")
                shutil.copy2(self.config["config_path"], stage / "config.yaml")
                self._write_inference_artifact(stage)
                self._write_inference_artifact(ema_dir)
                manifest = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "created_by": CHECKPOINT_CREATOR,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "global_step": global_step,
                    "world_size": get_world_size(),
                    "initialization": {
                        "checkpoint_dir": str(self.model.initialization_path),
                    },
                    "contains": ["model", "optimizer", "lr_scheduler", "global_step", "ema", "rng_per_rank"],
                    "contract": contract,
                }
                _write_json(stage / "manifest.json", manifest)
                _write_json(stage / "_SUCCESS", {"global_step": global_step, "created_at": manifest["created_at"]})
                os.rename(stage, final)
                logger.info("[openpi:checkpoint] saved complete checkpoint {}", final)
                _prune_checkpoints(
                    self.output_dir,
                    None if self.save_total_limit is None else int(self.save_total_limit),
                    self.keep_period,
                )
            except Exception as error:
                save_error = error
                save_error_message = f"{type(error).__name__}: {error}"
                if stage is not None and stage.exists():
                    try:
                        shutil.rmtree(stage)
                    except Exception as cleanup_error:
                        logger.error("[openpi:checkpoint] failed to remove staging directory {}: {}", stage, cleanup_error)
                        save_error_message += f"; staging cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}"

        if is_distributed():
            result = [save_error_message]
            dist.broadcast_object_list(result, src=0)
            if result[0] is not None:
                raise RuntimeError(f"Rank 0 failed to save checkpoint: {result[0]}") from save_error
        elif save_error is not None:
            raise save_error

    def _build_data_iterator(self, epoch: int, batches_in_epoch: int):
        self.dataloader.set_epoch(epoch)
        iterator = iter(self.dataloader)
        for _ in range(batches_in_epoch):
            try:
                next(iterator)
            except StopIteration as error:
                raise RuntimeError(f"Cannot restore data position epoch={epoch}, batch={batches_in_epoch}; dataset length changed") from error
        return iterator

    def _next_batch(self, iterator, epoch: int, batches_in_epoch: int):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            batches_in_epoch = 0
            self.dataloader.set_epoch(epoch)
            iterator = iter(self.dataloader)
            batch = next(iterator)
        return batch, iterator, epoch, batches_in_epoch + 1

    def train(self) -> None:
        _set_seed(int(self.training_config["seed"]))
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.training_config["gradient_checkpointing"]:
            self.model.enable_gradient_checkpointing()
        training_model = self._wrap_ddp()
        training_model.train()

        self.optimizer = torch.optim.AdamW(
            self.core_model.parameters(),
            lr=self.optimizer_spec["learning_rate"],
            betas=(self.optimizer_spec["adam_beta1"], self.optimizer_spec["adam_beta2"]),
            weight_decay=self.optimizer_spec["weight_decay"],
            eps=self.optimizer_spec["adam_epsilon"],
            foreach=False,
        )
        if not math.isclose(self.optimizer_spec["learning_rate"], float(self.schedule_spec["peak_lr"])):
            raise ValueError("optimizer.learning_rate must equal lr_schedule.peak_lr")
        self.lr_scheduler = _build_scheduler(self.optimizer, self.schedule_spec)
        self.ema = ExponentialMovingAverage(self.core_model, self.ema_decay)

        resume_path = self._resolve_resume_path()
        if resume_path is None:
            global_step, data_epoch, batches_in_epoch = 0, 0, 0
            logger.info("[openpi:init] model-only warm start; optimizer, scheduler, step, and EMA are new")
        else:
            global_step, data_epoch, batches_in_epoch = self._load_full_checkpoint(resume_path)
        if global_step > self.max_train_iters:
            raise ValueError(f"Checkpoint step {global_step} exceeds max_train_iters={self.max_train_iters}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        iterator = self._build_data_iterator(data_epoch, batches_in_epoch)
        if resume_path is not None:
            _restore_rng_state(self._resume_rng_state)
        self.optimizer.zero_grad(set_to_none=True)
        accumulated_loss: torch.Tensor | None = None
        numerical_state_validated = resume_path is not None
        last_saved_step = global_step if resume_path is not None else -1
        logger.info(
            "[openpi:train] start step={}/{} world_size={} grad_accum={} global_batch={} ema_decay={}",
            global_step,
            self.max_train_iters,
            get_world_size(),
            self.gradient_accumulation_iters,
            self.dataloader.metadata["global_batch_size"],
            self.ema_decay,
        )

        try:
            while global_step < self.max_train_iters:
                for accumulation_index in range(self.gradient_accumulation_iters):
                    batch, iterator, data_epoch, batches_in_epoch = self._next_batch(iterator, data_epoch, batches_in_epoch)
                    observation, actions = batch
                    observation = observation.to(self.model.device, non_blocking=True)
                    actions = actions.to(self.model.device, dtype=torch.float32, non_blocking=True)
                    sync_gradients = accumulation_index + 1 == self.gradient_accumulation_iters
                    sync_context = nullcontext() if sync_gradients or not isinstance(training_model, DistributedDataParallel) else training_model.no_sync()
                    with sync_context:
                        loss = training_model(observation, actions).mean()
                        if loss.dtype != torch.float32:
                            raise RuntimeError(f"OpenPI flow-matching loss must be float32, got {loss.dtype}")
                        (loss / self.gradient_accumulation_iters).backward()
                    detached_loss = loss.detach() / self.gradient_accumulation_iters
                    accumulated_loss = detached_loss if accumulated_loss is None else accumulated_loss + detached_loss

                if not numerical_state_validated:
                    _assert_fp32_gradients(self.core_model)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.core_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if not numerical_state_validated:
                    _assert_fp32_optimizer_state(self.optimizer)
                    numerical_state_validated = True
                    logger.info(
                        "[openpi:numerics] verified fp32 parameters, gradients, AdamW state, loss, and EMA masters; transformer compute={}",
                        self._checkpoint_contract()["numerical_policy"]["compute_dtype"],
                    )
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.core_model)
                global_step += 1

                if global_step == 1 or global_step % self.log_every_iters == 0 or global_step == self.max_train_iters:
                    loss_value = reduce_mean(accumulated_loss)
                    grad_value = reduce_mean(grad_norm.detach())
                    if torch.is_tensor(loss_value):
                        loss_value = loss_value.item()
                    if torch.is_tensor(grad_value):
                        grad_value = grad_value.item()
                    learning_rate = self.lr_scheduler.get_last_lr()[0]
                    logger.info(
                        "[openpi:train] step={}/{} loss={:.6f} grad_norm={:.6f} lr={:.8f}",
                        global_step,
                        self.max_train_iters,
                        loss_value,
                        grad_value,
                        learning_rate,
                    )
                    self.monitor.log_metrics(
                        {
                            "train/loss": loss_value,
                            "train/grad_norm": grad_value,
                            "train/lr": learning_rate,
                        },
                        step=global_step,
                    )
                accumulated_loss = None

                if self.save_every_iters and global_step % self.save_every_iters == 0:
                    self._save_checkpoint(global_step, data_epoch, batches_in_epoch)
                    last_saved_step = global_step

            if global_step != last_saved_step:
                self._save_checkpoint(global_step, data_epoch, batches_in_epoch)
            logger.info("[openpi:train] finished at step={}", global_step)
        finally:
            self.monitor.finish()
