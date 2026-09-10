"""Model-agnostic dataset for caches produced by ``cache_data.py``."""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from lightx2v_train.runtime.distributed import get_data_parallel_rank, get_data_parallel_world_size
from lightx2v_train.utils.generation_shapes import (
    GenerationShapeSampler,
    generation_shape_key,
    parse_generation_shapes,
    split_generation_shape_index,
)
from lightx2v_train.utils.registry import DATA_REGISTER


class CacheDataset(Dataset):
    """Load the common ``cache_data.jsonl`` format for every model family."""

    uses_cache_dataset = True

    def __init__(
        self,
        metadata_paths,
        prompt_dropout_rate=0.0,
        unconditional_prompt=" ",
        sample_processor=None,
        train_or_val="train",
        generation_shapes=None,
    ):
        self.prompt_dropout_rate = float(prompt_dropout_rate)
        self.unconditional_prompt = unconditional_prompt
        self.sample_processor = sample_processor
        self.train_or_val = train_or_val
        parsed_generation_shapes = parse_generation_shapes(generation_shapes) if generation_shapes is not None else []
        self.generation_shapes = tuple(shape.value for shape in parsed_generation_shapes)
        self.generation_shape_keys = tuple(generation_shape_key(shape) for shape in self.generation_shapes)
        self.has_multiple_generation_shapes = len(self.generation_shapes) > 1
        self.samples = []
        for metadata_path in metadata_paths:
            self.samples.extend(self._read_manifest(Path(metadata_path)))
        if not self.samples:
            raise ValueError(f"No training-cache records found in {metadata_paths}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        index, generation_shape = split_generation_shape_index(index)
        record = self.samples[index]
        if self.has_multiple_generation_shapes:
            if generation_shape is None:
                raise RuntimeError("Multiple generation shapes require GenerationShapeSampler for cache_dataset.")
            generation_shape = tuple(int(dimension) for dimension in generation_shape)
            shape_key = generation_shape_key(generation_shape)
            cache_path = record["training_caches"].get(shape_key)
            if cache_path is None:
                raise KeyError(f"No {shape_key} training cache is available for prompt {record['prompt']!r}.")
        else:
            cache_path = record["training_cache"]

        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        self._validate_cache(cache, cache_path, record["prompt"])
        if generation_shape is not None:
            cache["generation_shape"] = generation_shape

        conditioning = cache["conditioning"]
        use_unconditional = self.train_or_val == "train" and random.random() < self.prompt_dropout_rate
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
                record_path = f"{metadata_path}:{line_number}"
                if prompt is None:
                    raise ValueError(f"Training-cache record {record_path} must include prompt.")

                if self.has_multiple_generation_shapes:
                    cache_paths = record.get("training_caches")
                    if not isinstance(cache_paths, dict):
                        raise ValueError(f"Training-cache record {record_path} must include a training_caches mapping; rebuild the cache.")
                    actual_keys = set(cache_paths)
                    expected_keys = set(self.generation_shape_keys)
                    if actual_keys != expected_keys:
                        missing = sorted(expected_keys - actual_keys)
                        extra = sorted(actual_keys - expected_keys)
                        raise ValueError(f"Training-cache record {record_path} does not match configured generation shapes: missing={missing}, extra={extra}. Rebuild the cache.")
                    resolved_paths = {}
                    for shape_key in self.generation_shape_keys:
                        cache_path = cache_paths[shape_key]
                        if not str(cache_path).strip():
                            raise ValueError(f"Training-cache record {record_path} has an empty path for shape {shape_key}.")
                        cache_path = Path(cache_path)
                        if not cache_path.is_absolute():
                            cache_path = metadata_path.parent / cache_path
                        resolved_paths[shape_key] = cache_path
                    records.append({"prompt": str(prompt), "training_caches": resolved_paths})
                    continue

                cache_path = record.get("training_cache")
                if cache_path is None or not str(cache_path).strip():
                    raise ValueError(f"Training-cache record {record_path} must include training_cache.")
                cache_path = Path(cache_path)
                if not cache_path.is_absolute():
                    cache_path = metadata_path.parent / cache_path
                records.append({"prompt": str(prompt), "training_cache": cache_path})
        return records

    def _validate_cache(self, cache, path, prompt):
        required = {"inputs", "conditioning", "meta"}
        if not isinstance(cache, dict) or not required.issubset(cache):
            raise ValueError(f"Invalid training cache at {path}: expected inputs, conditioning, and meta mappings.")
        if not all(isinstance(cache[key], dict) for key in required):
            raise ValueError(f"Invalid training cache at {path}: inputs, conditioning, and meta must be mappings.")

        conditioning = cache["conditioning"]
        if "positive" not in conditioning:
            raise ValueError(f"Invalid training cache at {path}: conditioning.positive is missing.")
        if conditioning.get("prompt") != prompt:
            raise ValueError(f"Prompt in {path} does not match cache_data.jsonl. Rebuild the training cache.")


@DATA_REGISTER("cache_dataset")
def build_cache_dataset(
    data_config_split,
    train_or_val="train",
    unconditional_prompt=" ",
    sample_processor=None,
):
    if train_or_val not in {"train", "val"}:
        raise ValueError(f"cache_dataset only supports train or val, got {train_or_val!r}.")
    data_paths = data_config_split["data_path"]
    if isinstance(data_paths, (str, Path)):
        data_paths = [data_paths]

    dataset = CacheDataset(
        metadata_paths=data_paths,
        prompt_dropout_rate=data_config_split.get("prompt_dropout_rate", 0.0),
        unconditional_prompt=unconditional_prompt,
        sample_processor=sample_processor,
        train_or_val=train_or_val,
        generation_shapes=data_config_split.get("generation_shapes"),
    )
    world_size = get_data_parallel_world_size()
    shuffle = data_config_split.get("shuffle", train_or_val == "train")
    drop_last = data_config_split.get("drop_last", False)
    sampler = None
    if train_or_val == "train" and dataset.has_multiple_generation_shapes:
        sampler = GenerationShapeSampler(
            dataset,
            num_replicas=world_size,
            rank=get_data_parallel_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
            generation_shapes=data_config_split["generation_shapes"],
        )
        shuffle = False
        drop_last = False
    elif world_size > 1 and train_or_val == "train":
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=get_data_parallel_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False
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
        drop_last=drop_last,
        collate_fn=_single_sample_collate,
        **loader_kwargs,
    )


def _single_sample_collate(samples):
    if len(samples) != 1:
        raise ValueError(f"Cached data requires batch_size=1, got {len(samples)} samples.")
    return samples[0]
