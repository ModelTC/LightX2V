"""Model-agnostic dataset for caches produced by ``cache_data.py``."""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from lightx2v_train.data.training_cache import CACHE_SCHEMA_VERSION
from lightx2v_train.runtime.distributed import get_data_parallel_rank, get_data_parallel_world_size
from lightx2v_train.utils.registry import DATA_REGISTER


class TrainingCacheDataset(Dataset):
    """Load the common ``cache_data.jsonl`` format for every model family."""

    def __init__(
        self,
        metadata_paths,
        prompt_dropout_rate=0.0,
        unconditional_prompt=" ",
        expected_cache_info=None,
    ):
        self.prompt_dropout_rate = float(prompt_dropout_rate)
        self.unconditional_prompt = unconditional_prompt
        self.expected_cache_info = expected_cache_info
        self.samples = []
        for metadata_path in metadata_paths:
            self.samples.extend(self._read_manifest(Path(metadata_path)))
        if not self.samples:
            raise ValueError(f"No training-cache records found in {metadata_paths}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        cache_path = record["training_cache"]
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        self._validate_cache(cache, cache_path, record["prompt"])

        conditioning = cache["conditioning"]
        use_unconditional = random.random() < self.prompt_dropout_rate
        if use_unconditional and "unconditional" not in conditioning:
            raise ValueError(f"Training cache {cache_path} has no unconditional condition for prompt dropout.")
        conditioning["active"] = "unconditional" if use_unconditional else "positive"
        conditioning["prompt"] = conditioning.get("unconditional_prompt", self.unconditional_prompt) if use_unconditional else record["prompt"]
        cache["meta"]["training_cache_path"] = str(cache_path)
        return cache

    def _read_manifest(self, metadata_path):
        if metadata_path.suffix.lower() != ".jsonl":
            raise ValueError(f"Training-cache manifests must be .jsonl files, got: {metadata_path}")
        records = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt = record.get("prompt")
                cache_path = record.get("training_cache")
                if prompt is None or cache_path is None or not str(cache_path).strip():
                    raise ValueError(f"Training-cache record {metadata_path}:{line_number} must include prompt and training_cache.")
                cache_path = Path(cache_path)
                if not cache_path.is_absolute():
                    cache_path = metadata_path.parent / cache_path
                records.append(
                    {
                        "prompt": str(prompt),
                        "training_cache": cache_path,
                    }
                )
        return records

    def _validate_cache(self, cache, path, prompt):
        required = {"inputs", "conditioning", "meta"}
        if not isinstance(cache, dict) or not required.issubset(cache):
            raise ValueError(f"Invalid training cache at {path}: expected inputs, conditioning, and meta mappings.")
        if not all(isinstance(cache[key], dict) for key in required):
            raise ValueError(f"Invalid training cache at {path}: inputs, conditioning, and meta must be mappings.")

        cache_info = cache.get("cache_info")
        if not isinstance(cache_info, dict):
            raise ValueError(f"Invalid training cache metadata at {path}.")
        if cache_info.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported training cache schema at {path}: {cache_info.get('schema_version')!r}.")
        if self.expected_cache_info is not None:
            for key, expected in self.expected_cache_info.items():
                if cache_info.get(key) != expected:
                    raise ValueError(f"Training cache {path} has incompatible {key}: expected {expected!r}, got {cache_info.get(key)!r}.")

        conditioning = cache["conditioning"]
        if "positive" not in conditioning:
            raise ValueError(f"Invalid training cache at {path}: conditioning.positive is missing.")
        if conditioning.get("prompt") != prompt:
            raise ValueError(f"Prompt in {path} does not match cache_data.jsonl. Rebuild the training cache.")


@DATA_REGISTER("training_cache_dataset")
def build_training_cache_dataset(
    data_config_split,
    train_or_val="train",
    unconditional_prompt=" ",
    expected_cache_info=None,
):
    if train_or_val != "train":
        raise ValueError("training_cache_dataset is only valid for the training split.")
    data_paths = data_config_split["data_path"]
    if isinstance(data_paths, (str, Path)):
        data_paths = [data_paths]

    dataset = TrainingCacheDataset(
        metadata_paths=data_paths,
        prompt_dropout_rate=data_config_split.get("prompt_dropout_rate", 0.0),
        unconditional_prompt=unconditional_prompt,
        expected_cache_info=expected_cache_info,
    )
    world_size = get_data_parallel_world_size()
    shuffle = data_config_split.get("shuffle", True)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=get_data_parallel_rank(),
            shuffle=shuffle,
            drop_last=data_config_split.get("drop_last", False),
        )
        if world_size > 1
        else None
    )
    num_workers = int(data_config_split.get("num_workers", 8))
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = data_config_split.get("persistent_workers", True)
        loader_kwargs["prefetch_factor"] = data_config_split.get("prefetch_factor", 2)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=data_config_split.get("pin_memory", True),
        drop_last=data_config_split.get("drop_last", False),
        collate_fn=_single_sample_collate,
        **loader_kwargs,
    )


def _single_sample_collate(samples):
    if len(samples) != 1:
        raise ValueError(f"Cached training requires batch_size=1, got {len(samples)} samples.")
    return samples[0]
