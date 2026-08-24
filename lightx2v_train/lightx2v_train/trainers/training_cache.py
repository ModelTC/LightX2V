import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_train.data.training_cache import preserve_cache_dtype, training_cache_info
from lightx2v_train.data.utils import require_singleton_dataloader
from lightx2v_train.model_capabilities import (
    ConsistencyModelCapability,
    DistributionMatchingCapability,
    FlowMatchingSFTCapability,
    TeacherForcingCapability,
)
from lightx2v_train.runtime.distributed import get_rank, get_sequence_parallel_rank, is_distributed, is_main_process
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


def _dataset_index(sample):
    value = sample["meta"].pop("dataset_index")
    return int(value.item() if torch.is_tensor(value) else value)


@TRAINER_REGISTER("training_cache")
class TrainingCacheTrainer:
    def __init__(self, config):
        self.config = config
        self.cache_config = config["training_cache"]
        self.training_method = config["training"]["method"]
        self.cache_info = {
            **training_cache_info(config),
            "storage_dtype": self.cache_config["save_dtype"],
        }

    def set_model(self, model):
        capability_type = CACHE_CAPABILITIES.get(self.training_method)
        if capability_type is None:
            supported = ", ".join(sorted(CACHE_CAPABILITIES))
            raise ValueError(f"Training cache does not support {self.training_method!r}; expected one of: {supported}.")
        self.model = model
        self.encoder = model.ensure_capabilities().require(capability_type)

    def set_data(self, dataloader_train, dataloader_eval=None):
        del dataloader_eval
        require_singleton_dataloader(dataloader_train, "Cache dataloader")
        self.dataloader = dataloader_train

    def _encode(self, sample, dtype):
        encoded = self.encoder.encode_training_cache(sample)
        cache = {
            "cache_info": self.cache_info,
            **encoded,
        }
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
        if cache.get("cache_info") != self.cache_info:
            raise ValueError(f"Training cache at {path} is incompatible with the current configuration. Rebuild it.")
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
            raise ValueError("Cache construction requires data.train.dataset_repeat=1.")
        dtype = CACHE_DTYPES[self.cache_config["save_dtype"]]
        records = []

        for sample in self.dataloader:
            index = _dataset_index(sample)
            if get_sequence_parallel_rank() != 0 or index >= sample_count:
                continue

            record = dataset.cache_source_record(index)
            prompt = dataset.cache_source_prompt(index)
            sample["conditioning"]["prompt"] = prompt
            record["prompt"] = prompt
            cache_path = cache_dir / f"{index:08d}.pt"
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

            record["training_cache"] = cache_path.relative_to(output_dir).as_posix()
            records.append((index, record))
            logger.info("[cache] {}/{} -> {}", index + 1, sample_count, cache_path)

        records = self._gather_records(records)
        if is_main_process():
            indexed_records = {index: record for index, record in records}
            if sorted(indexed_records) != list(range(sample_count)):
                raise RuntimeError("Training cache is incomplete.")
            _write_jsonl([indexed_records[index] for index in range(sample_count)], output_dir / "cache_data.jsonl")
            logger.info("[cache] wrote {} samples to {}", sample_count, output_dir / "cache_data.jsonl")
