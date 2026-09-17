"""Local LeRobot input pipeline for OpenPI pi0.5-LIBERO training."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece
import torch
from PIL import Image
from loguru import logger

from lightx2v_train.runtime.distributed import get_rank, get_world_size, is_distributed
from lightx2v_train.utils.registry import DATA_REGISTER

LIBERO_STATE_DIM = 8
LIBERO_ACTION_DIM = 7
MODEL_IMAGE_SIZE = 224

IMAGE_PIPELINE_CONTRACT = {
    "target_size": MODEL_IMAGE_SIZE,
    "resize": "openpi_client_pil_bilinear_uint8_resize_with_pad",
    "quantization": "float01_to_uint8_truncate_before_resize",
    "augmentation": "openpi_augmax_0.4.1_per_sample",
}


@dataclass
class OpenPIObservation:
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None

    def to(self, device: torch.device, *, non_blocking: bool = False) -> "OpenPIObservation":
        def move(value):
            return None if value is None else value.to(device, non_blocking=non_blocking)

        return OpenPIObservation(
            images={name: move(image) for name, image in self.images.items()},
            image_masks={name: move(mask) for name, mask in self.image_masks.items()},
            state=move(self.state),
            tokenized_prompt=move(self.tokenized_prompt),
            tokenized_prompt_mask=move(self.tokenized_prompt_mask),
            token_ar_mask=move(self.token_ar_mask),
            token_loss_mask=move(self.token_loss_mask),
        )

    def pin_memory(self) -> "OpenPIObservation":
        def pin(value):
            return None if value is None else value.pin_memory()

        return OpenPIObservation(
            images={name: pin(image) for name, image in self.images.items()},
            image_masks={name: pin(mask) for name, mask in self.image_masks.items()},
            state=pin(self.state),
            tokenized_prompt=pin(self.tokenized_prompt),
            tokenized_prompt_mask=pin(self.tokenized_prompt_mask),
            token_ar_mask=pin(self.token_ar_mask),
            token_loss_mask=pin(self.token_loss_mask),
        )


class _PaligemmaTokenizer:
    def __init__(self, path: Path, max_token_len: int):
        if not path.is_file():
            raise FileNotFoundError(f"PaliGemma tokenizer not found: {path}")
        self.max_token_len = max_token_len
        self.processor = sentencepiece.SentencePieceProcessor(model_proto=path.read_bytes())

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        prompt = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = self.processor.encode(prompt, add_bos=True) + self.processor.encode("\n")
        if len(tokens) > self.max_token_len:
            logger.warning("Prompt has {} tokens and will be truncated to {}", len(tokens), self.max_token_len)
        tokens = tokens[: self.max_token_len]
        mask = [True] * len(tokens)
        padding = self.max_token_len - len(tokens)
        tokens.extend([0] * padding)
        mask.extend([False] * padding)
        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=np.bool_)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_quantile_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"OpenPI normalization statistics not found: {path}")
    with path.open(encoding="utf-8") as stream:
        stats = json.load(stream)["norm_stats"]
    result = {}
    for key, expected_dim in (("state", LIBERO_STATE_DIM), ("actions", LIBERO_ACTION_DIM)):
        if key not in stats:
            raise ValueError(f"Normalization statistics have no {key!r} entry: {path}")
        values = {name: np.asarray(value) for name, value in stats[key].items()}
        for name in ("q01", "q99"):
            if name not in values or values[name].shape[-1] < expected_dim:
                raise ValueError(f"Invalid {key}.{name} in {path}: expected at least {expected_dim} values")
            if not np.isfinite(values[name][..., :expected_dim]).all():
                raise ValueError(f"Non-finite values in {key}.{name}: {path}")
        if np.any(values["q99"][..., :expected_dim] < values["q01"][..., :expected_dim]):
            raise ValueError(f"Every {key}.q99 value must be greater than or equal to q01: {path}")
        result[key] = values
    return result


def _normalize_quantile(value: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    q01 = stats["q01"][..., : value.shape[-1]]
    q99 = stats["q99"][..., : value.shape[-1]]
    return ((value - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).astype(np.float32)


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_as_uint8_hwc(value: Any, key: str) -> np.ndarray:
    image = _as_numpy(value)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{key} must be an RGB image, got shape={image.shape}, dtype={image.dtype}")
    return np.ascontiguousarray(image)


def _resize_with_pad_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    current_height, current_width = image.shape[:2]
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = Image.fromarray(image).resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    canvas.paste(resized, ((width - resized_width) // 2, (height - resized_height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def _pad_last_dim(value: np.ndarray, size: int) -> np.ndarray:
    if value.shape[-1] > size:
        raise ValueError(f"Cannot pad dimension {value.shape[-1]} to the smaller size {size}")
    if value.shape[-1] == size:
        return value
    widths = [(0, 0)] * value.ndim
    widths[-1] = (0, size - value.shape[-1])
    return np.pad(value, widths)


class OpenPILiberoDataset(torch.utils.data.Dataset):
    """Apply the official LIBERO repack/normalize/tokenize/pad contract locally."""

    def __init__(
        self,
        dataset,
        tasks: dict[int, str],
        norm_stats_path: Path,
        tokenizer_path: Path,
        *,
        action_horizon: int,
        action_dim: int,
        max_token_len: int,
    ):
        self.dataset = dataset
        self.tasks = {int(index): prompt for index, prompt in tasks.items()}
        self.norm_stats = _load_quantile_stats(norm_stats_path)
        self.tokenizer = _PaligemmaTokenizer(tokenizer_path, max_token_len)
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        task_index = int(_as_numpy(item["task_index"]).reshape(()))
        prompt = self.tasks[task_index]

        base_image = _resize_with_pad_uint8(_image_as_uint8_hwc(item["image"], "image"), MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
        wrist_image = _resize_with_pad_uint8(_image_as_uint8_hwc(item["wrist_image"], "wrist_image"), MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
        state = _as_numpy(item["state"]).astype(np.float32, copy=False)
        actions = _as_numpy(item["actions"]).astype(np.float32, copy=False)
        if state.shape != (LIBERO_STATE_DIM,):
            raise ValueError(f"state must have shape ({LIBERO_STATE_DIM},), got {state.shape}")
        if actions.shape != (self.action_horizon, LIBERO_ACTION_DIM):
            raise ValueError(f"actions must have shape ({self.action_horizon}, {LIBERO_ACTION_DIM}), got {actions.shape}")

        state = _pad_last_dim(_normalize_quantile(state, self.norm_stats["state"]), self.action_dim)
        actions = _pad_last_dim(_normalize_quantile(actions, self.norm_stats["actions"]), self.action_dim)
        tokens, token_mask = self.tokenizer.tokenize(prompt)
        return {
            "images": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_masks": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": False,
            },
            "state": state,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "actions": actions,
        }


def _collate(items: list[dict[str, Any]]) -> tuple[OpenPIObservation, torch.Tensor]:
    image_names = tuple(items[0]["images"])
    images = {}
    masks = {}
    for name in image_names:
        batch = torch.from_numpy(np.stack([item["images"][name] for item in items]))
        images[name] = batch.permute(0, 3, 1, 2).to(torch.float32).div_(127.5).sub_(1.0)
        masks[name] = torch.as_tensor([item["image_masks"][name] for item in items], dtype=torch.bool)

    observation = OpenPIObservation(
        images=images,
        image_masks=masks,
        state=torch.from_numpy(np.stack([item["state"] for item in items])),
        tokenized_prompt=torch.from_numpy(np.stack([item["tokenized_prompt"] for item in items])),
        tokenized_prompt_mask=torch.from_numpy(np.stack([item["tokenized_prompt_mask"] for item in items])),
    )
    actions = torch.from_numpy(np.stack([item["actions"] for item in items]))
    return observation, actions


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class OpenPIDataLoader:
    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        generator: torch.Generator,
        seed: int,
        batches_per_epoch: int,
        metadata: dict[str, Any],
    ):
        self.loader = loader
        self.generator = generator
        self.seed = seed
        self.batches_per_epoch = batches_per_epoch
        self.metadata = metadata

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(self.batches_per_epoch):
            yield next(iterator)

    def set_epoch(self, epoch: int) -> None:
        self.generator.manual_seed(self.seed + epoch)
        if hasattr(self.loader.sampler, "set_epoch"):
            self.loader.sampler.set_epoch(epoch)


@DATA_REGISTER("openpi_libero")
def build_openpi_libero(config: dict[str, Any], train_or_val: str):
    if train_or_val != "train":
        raise ValueError("OpenPI LIBERO integration currently provides a training split only")

    dataset_root = Path(config["root"]).expanduser().resolve()
    norm_stats_path = Path(config["norm_stats_path"]).expanduser().resolve()
    tokenizer_path = Path(config["tokenizer_path"]).expanduser().resolve()
    cache_dir = Path(config["hf_cache_dir"]).expanduser().resolve()
    if not (dataset_root / "meta/info.json").is_file():
        raise FileNotFoundError(f"LeRobot dataset is incomplete or missing: {dataset_root}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    repo_id = str(config["repo_id"])
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    action_horizon = int(config["action_horizon"])
    delta_timestamps = {"actions": [step / metadata.fps for step in range(action_horizon)]}
    source = LeRobotDataset(
        repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )
    dataset = OpenPILiberoDataset(
        source,
        metadata.tasks,
        norm_stats_path,
        tokenizer_path,
        action_horizon=action_horizon,
        action_dim=int(config["action_dim"]),
        max_token_len=int(config["max_token_len"]),
    )

    global_batch_size = int(config["global_batch_size"])
    accumulation = int(config["gradient_accumulation_iters"])
    divisor = get_world_size() * accumulation
    if global_batch_size % divisor:
        raise ValueError(f"data.train.global_batch_size={global_batch_size} must be divisible by world_size * gradient_accumulation_iters={divisor}")
    batch_size = global_batch_size // divisor
    if batch_size < 1:
        raise ValueError("Per-rank micro batch size must be at least one")

    sampler = None
    shuffle = bool(config["shuffle"])
    seed = int(config["seed"])
    if is_distributed():
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )
        shuffle = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    num_workers = int(config["num_workers"])
    multiprocessing_options = {"multiprocessing_context": "spawn"} if num_workers > 0 else {}
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=bool(config["pin_memory"]),
        drop_last=True,
        collate_fn=_collate,
        worker_init_fn=_seed_worker,
        generator=generator,
        **multiprocessing_options,
    )
    optimizer_batches_per_epoch = len(dataset) // global_batch_size
    batches_per_epoch = optimizer_batches_per_epoch * accumulation
    if batches_per_epoch < 1 or batches_per_epoch > len(loader):
        raise RuntimeError(
            f"Cannot form a complete optimizer batch from the dataset: dataset={len(dataset)}, global_batch={global_batch_size}, micro_batches={len(loader)}, required={batches_per_epoch}"
        )
    logger.info(
        "[openpi:data] root={} episodes={} frames={} language_tasks={} fps={} per_rank_batch={} world_size={} grad_accum={} global_batch={}",
        dataset_root,
        metadata.total_episodes,
        metadata.total_frames,
        metadata.total_tasks,
        metadata.fps,
        batch_size,
        get_world_size(),
        accumulation,
        global_batch_size,
    )
    return OpenPIDataLoader(
        loader,
        generator,
        seed,
        batches_per_epoch,
        {
            "root": str(dataset_root),
            "repo_id": repo_id,
            "norm_stats_path": str(norm_stats_path),
            "tokenizer_path": str(tokenizer_path),
            "episodes": metadata.total_episodes,
            "frames": metadata.total_frames,
            "language_tasks": metadata.total_tasks,
            "fps": metadata.fps,
            "global_batch_size": global_batch_size,
            "per_rank_batch_size": batch_size,
            "optimizer_batches_per_epoch": optimizer_batches_per_epoch,
            "micro_batches_per_epoch": batches_per_epoch,
            "shuffle": bool(config["shuffle"]),
            "seed": seed,
            "dataset_info_sha256": _sha256(dataset_root / "meta/info.json"),
            "norm_stats_sha256": _sha256(norm_stats_path),
            "tokenizer_sha256": _sha256(tokenizer_path),
            "image_pipeline": IMAGE_PIPELINE_CONTRACT,
        },
    )
