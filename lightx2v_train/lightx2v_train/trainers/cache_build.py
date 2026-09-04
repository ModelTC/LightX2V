import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_train.data.utils import preserve_cache_dtype, require_singleton_dataloader
from lightx2v_train.model_capabilities import (
    ConsistencyModelCapability,
    DistributionMatchingCapability,
    FlowMatchingSFTCapability,
    TeacherForcingCapability,
)
from lightx2v_train.runtime.distributed import get_rank, get_sequence_parallel_rank, is_distributed, is_main_process
from lightx2v_train.utils.generation_shapes import generation_shape_key, parse_generation_shapes
from lightx2v_train.utils.registry import TRAINER_REGISTER

CACHE_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
CACHE_CAPABILITIES = {
    "flow_matching": FlowMatchingSFTCapability,
    "consistency": ConsistencyModelCapability,
    "dmd": DistributionMatchingCapability,
    "autoregressive_dmd": DistributionMatchingCapability,
    "phased_dmd": DistributionMatchingCapability,
    "sgmd": DistributionMatchingCapability,
    "teacher_forcing": TeacherForcingCapability,
}


def _to_cpu(value, dtype, key=None):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        return value.to(dtype) if value.is_floating_point() and not preserve_cache_dtype(key) else value
    if isinstance(value, dict):
        return {name: _to_cpu(item, dtype, name) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(item, dtype, key) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item, dtype, key) for item in value]
    return value


def _atomic_save(value, path):
    temporary = path.with_suffix(f"{path.suffix}.rank{get_rank():05d}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_jsonl(records, path):
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _write_json(value, path):
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _dataset_index(sample):
    value = sample["meta"].pop("dataset_index")
    return int(value.item() if torch.is_tensor(value) else value)


@TRAINER_REGISTER("cache_build")
class CacheBuildTrainer:
    def __init__(self, config):
        self.config = config
        self.cache_config = config["cache_build"]
        self.data_split = self.cache_config.get("data_split", "train")
        self.training_method = config["training"]["method"]
        generation_shapes = config.get("training", {}).get("dmd", {}).get("generation_shapes")
        parsed_generation_shapes = parse_generation_shapes(generation_shapes) if self.data_split == "train" and generation_shapes is not None else []
        self.generation_shapes = tuple(shape.value for shape in parsed_generation_shapes)
        self.cache_generation_shapes = self.generation_shapes if len(self.generation_shapes) > 1 else ()

    def set_model(self, model):
        capability_type = CACHE_CAPABILITIES.get(self.training_method)
        if capability_type is None:
            supported = ", ".join(sorted(CACHE_CAPABILITIES))
            raise ValueError(f"Cache build does not support {self.training_method!r}; expected one of: {supported}.")
        self.model = model
        self.encoder = model.ensure_capabilities().require(capability_type)

    def set_data(self, dataloader_train, dataloader_val=None):
        del dataloader_val
        require_singleton_dataloader(dataloader_train, "Cache dataloader")
        self.dataloader = dataloader_train

    def _encode(self, sample, dtype):
        cache = self.encoder.encode_training_cache(sample)
        cache.pop("generation_shape", None)
        self._validate(
            cache,
            sample["conditioning"]["prompt"],
            source_inputs=sample.get("inputs"),
        )
        return _to_cpu(cache, dtype)

    def _validate(self, cache, prompt, path="<encoded cache>", source_inputs=None):
        required = {"inputs", "conditioning", "meta"}
        if not isinstance(cache, dict) or not required.issubset(cache):
            raise ValueError(f"Invalid training cache at {path}: expected inputs, conditioning, and meta mappings.")
        if not all(isinstance(cache[key], dict) for key in required):
            raise ValueError(f"Invalid training cache at {path}: inputs, conditioning, and meta must be mappings.")
        if source_inputs and not cache["inputs"]:
            raise ValueError(f"Training cache at {path} has no encoded model inputs for a source sample that contains inputs. Rebuild it with --overwrite.")
        conditioning = cache["conditioning"]
        if not isinstance(conditioning, dict) or "positive" not in conditioning:
            raise ValueError(f"Invalid training cache at {path}: conditioning.positive is missing.")
        if conditioning.get("prompt") != prompt:
            raise ValueError(f"Training cache at {path} has a different prompt. Rebuild it.")

    def _gather_records(self, records):
        if not is_distributed():
            return records
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, records)
        return [item for rank_records in gathered for item in rank_records]

    @torch.inference_mode()
    def train(self):
        output_dir = Path(self.cache_config["output_dir"]).resolve()
        cache_dir = output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        dataset = self.dataloader.dataset
        if not hasattr(dataset, "cache_source_record") or not hasattr(dataset, "cache_source_prompt"):
            raise TypeError(f"{type(dataset).__name__} cannot be used as a cache source; use image_dataset, video_dataset, or prompt_dataset.")
        sample_count = len(dataset)
        if len(getattr(dataset, "samples", ())) != sample_count:
            raise ValueError(f"Cache construction requires data.{self.data_split}.dataset_repeat=1.")
        dtype = CACHE_DTYPES[self.cache_config["save_dtype"]]
        records = []
        cache_variants = self.cache_generation_shapes or (None,)

        for sample in self.dataloader:
            index = _dataset_index(sample)
            if get_sequence_parallel_rank() != 0 or index >= sample_count:
                continue

            record = dataset.cache_source_record(index)
            prompt = dataset.cache_source_prompt(index)
            sample["conditioning"]["prompt"] = prompt
            record["prompt"] = prompt
            record.pop("training_cache", None)
            record.pop("training_caches", None)

            for generation_shape in cache_variants:
                sample.pop("generation_shape", None)
                shape_key = None
                if generation_shape is not None:
                    sample["generation_shape"] = generation_shape
                    shape_key = generation_shape_key(generation_shape)
                cache_name = f"{index:08d}-{shape_key}.pt" if shape_key is not None else f"{index:08d}.pt"
                cache_path = cache_dir / cache_name
                if self.cache_config["overwrite"] or not cache_path.exists():
                    torch.manual_seed(self.cache_config["seed"] + index)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(self.cache_config["seed"] + index)
                    _atomic_save(self._encode(sample, dtype), cache_path)
                else:
                    existing = torch.load(cache_path, map_location="cpu", weights_only=True)
                    self._validate(
                        existing,
                        sample["conditioning"]["prompt"],
                        cache_path,
                        source_inputs=sample.get("inputs"),
                    )

                relative_cache_path = cache_path.relative_to(output_dir).as_posix()
                records.append((index, shape_key, record, relative_cache_path))
                logger.info(
                    "[cache][{}] sample={}/{} shape={} -> {}",
                    self.data_split,
                    index + 1,
                    sample_count,
                    shape_key or "default",
                    cache_path,
                )

        records = self._gather_records(records)
        if is_main_process():
            indexed_records = {}
            indexed_cache_paths = {}
            for index, shape_key, record, cache_path in records:
                indexed_records.setdefault(index, record)
                cache_paths = indexed_cache_paths.setdefault(index, {})
                previous_path = cache_paths.setdefault(shape_key, cache_path)
                if previous_path != cache_path:
                    raise RuntimeError(f"Conflicting cache paths for sample {index}, shape {shape_key}: {previous_path} != {cache_path}.")

            if sorted(indexed_records) != list(range(sample_count)):
                raise RuntimeError("Training cache is incomplete.")
            if self.cache_generation_shapes:
                expected_shape_keys = tuple(generation_shape_key(shape) for shape in self.cache_generation_shapes)
                for index in range(sample_count):
                    cache_paths = indexed_cache_paths[index]
                    if set(cache_paths) != set(expected_shape_keys):
                        raise RuntimeError(f"Training cache for sample {index} is missing one or more generation shapes.")
                    indexed_records[index]["training_caches"] = {shape_key: cache_paths[shape_key] for shape_key in expected_shape_keys}
            else:
                for index in range(sample_count):
                    indexed_records[index]["training_cache"] = indexed_cache_paths[index][None]

            _write_jsonl([indexed_records[index] for index in range(sample_count)], output_dir / "cache_data.jsonl")
            cache_metadata = {
                "storage_dtype": self.cache_config["save_dtype"],
                "data_split": self.data_split,
            }
            _write_json(cache_metadata, output_dir / "cache_meta.json")
            logger.info("[cache][{}] wrote {} samples to {}", self.data_split, sample_count, output_dir / "cache_data.jsonl")
