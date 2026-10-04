import hashlib
import io
import math
from collections import Counter, OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from lightx2v_train.data.utils import (
    prompt_text,
    read_records,
    record_value,
    resolve_data_path,
    to_list,
)
from lightx2v_train.runtime.distributed import get_data_parallel_rank, get_data_parallel_world_size
from lightx2v_train.utils.registry import DATA_REGISTER

METADATA_SUFFIXES = {".jsonl", ".json", ".csv"}
PROMPT_SUFFIXES = {".txt", ".list"}
CONDITION_KEYS = (
    "prompt_embed",
    "video_prompt_embeds",
    "audio_prompt_embeds",
    "prompt_embeds",
    "text_token_tags",
    "prompt_attention_mask",
    "video_context",
    "audio_context",
    "context_mask",
)


def _strip_condition_batch(value):
    if torch.is_tensor(value):
        if value.ndim >= 2 and value.shape[0] == 1:
            return value[0]
        return value
    if isinstance(value, dict):
        return {key: _strip_condition_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_condition_batch(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_condition_batch(item) for item in value)
    return value


def _condition_payload(item):
    if isinstance(item, (list, tuple)):
        return _strip_condition_batch(item)
    if not isinstance(item, dict):
        raise TypeError(f"Condition payload must be a dict/list/tuple, got {type(item)!r}.")
    if "conditioning" in item:
        item = item["conditioning"]
    if "positive" in item:
        conditions = item["positive"]
    elif "conditions" in item:
        conditions = item["conditions"]
    elif "condition" in item:
        conditions = item["condition"]
    else:
        conditions = {key: item[key] for key in CONDITION_KEYS if key in item}
    if conditions is None or (isinstance(conditions, (dict, list, tuple)) and not conditions):
        raise KeyError("Condition payload must contain positive/conditions/condition or prompt embedding tensors.")
    return _strip_condition_batch(conditions)


def _video_latent_payload(data):
    if torch.is_tensor(data):
        return {"latents": data}
    if not isinstance(data, dict):
        return data
    normalized = dict(data)
    latents = normalized.get("latents")
    if not torch.is_tensor(latents) or latents.dim() != 2:
        return normalized
    num_frames = int(normalized["num_frames"])
    height = int(normalized["height"])
    width = int(normalized["width"])
    normalized["latents"] = latents.reshape(num_frames, height, width, latents.shape[-1]).permute(3, 0, 1, 2).contiguous()
    return normalized


def _metadata_path(path):
    path = Path(path)
    if path.is_dir():
        metadata_path = path / "metadata.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Dataset directory must contain metadata.jsonl: {path}")
        return metadata_path
    if path.suffix.lower() not in METADATA_SUFFIXES:
        raise ValueError(f"Metadata dataset path must be .jsonl/.json/.csv or a directory containing metadata.jsonl, got: {path}")
    return path


def _is_lmdb_path(path):
    path = Path(path)
    return path.is_dir() and ((path / "data.mdb").is_file() or (path / "lock.mdb").is_file())


def _resolve_required_path(value, base_dir, key):
    path = resolve_data_path(value, base_dir)
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"{key} points to a missing file: {path}")
    return path


class LatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_paths,
        dataset_repeat=1,
        max_samples=None,
        prompt_column="caption",
        prompt_index=0,
        negative_condition_path=None,
        defer_latent_loading=False,
    ):
        self.paths = [Path(path) for path in to_list(data_paths)]
        if not self.paths:
            raise ValueError("latent_dataset requires data_path.")
        self.dataset_repeat = int(dataset_repeat)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.prompt_column = prompt_column
        self.prompt_index = int(prompt_index)
        self.defer_latent_loading = bool(defer_latent_loading)
        self.samples = []
        self.lmdb_envs = []
        self.negative_condition = self._load_negative_condition(negative_condition_path)

        for path in self.paths:
            self._index_path(path)
        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]
        if not self.samples:
            raise RuntimeError(f"No usable latent samples found from data_path={data_paths}.")
        logger.info("[data] latent_dataset samples={} repeat={}", len(self.samples), self.dataset_repeat)

    def _index_path(self, path):
        if _is_lmdb_path(path):
            self._index_lmdb(path)
            return
        self._index_metadata(_metadata_path(path))

    def _load_negative_condition(self, negative_condition_path):
        if negative_condition_path is None:
            for path in self.paths:
                base = path if path.is_dir() else path.parent
                candidate = base / "negative_condition.pt"
                if candidate.is_file():
                    negative_condition_path = candidate
                    break
        if negative_condition_path is None:
            return None
        return _condition_payload(torch.load(negative_condition_path, map_location="cpu", weights_only=False))

    def _index_metadata(self, metadata_path):
        for row in read_records(metadata_path, prompt_column=self.prompt_column, prompt_index=self.prompt_index):
            self.samples.append({"type": "metadata", "row": row, "base_dir": str(metadata_path.parent)})

    def _index_lmdb(self, data_path):
        try:
            import lmdb
        except ImportError as error:
            raise ImportError("latent_dataset LMDB input requires the 'lmdb' Python package.") from error

        env = lmdb.open(str(data_path), readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            format_bytes = txn.get(b"__format__")
            sample_count_bytes = txn.get(b"sample_count") or txn.get(b"num_samples")

        env_index = len(self.lmdb_envs)
        if format_bytes is not None and format_bytes.decode() == "sample_pt":
            if sample_count_bytes is None:
                raise KeyError("sample_pt LMDB dataset requires sample_count.")
            sample_count = int(sample_count_bytes.decode())
            self.lmdb_envs.append({"env": env, "format": "sample_pt", "path": str(data_path)})
            for row_index in range(sample_count):
                self.samples.append({"type": "lmdb_sample", "env_index": env_index, "row_index": row_index})
            return

        latents_shape = self._get_lmdb_shape(env, "latents")
        self.lmdb_envs.append({"env": env, "format": "wan_latents", "latents_shape": latents_shape, "path": str(data_path)})
        for row_index in range(latents_shape[0]):
            self.samples.append({"type": "lmdb_wan", "env_index": env_index, "row_index": row_index})

    @staticmethod
    def _get_lmdb_shape(env, array_name):
        with env.begin() as txn:
            shape_bytes = txn.get(f"{array_name}_shape".encode())
        if shape_bytes is None:
            raise KeyError(f"{array_name}_shape not found in LMDB dataset.")
        return tuple(map(int, shape_bytes.decode().split()))

    @staticmethod
    def _retrieve_lmdb_row(env, array_name, dtype, row_index, shape=None):
        data_key = f"{array_name}_{row_index}_data".encode()
        with env.begin() as txn:
            row_bytes = txn.get(data_key)
        if row_bytes is None:
            raise KeyError(f"{data_key!r} not found in LMDB dataset.")
        if dtype is str:
            return row_bytes.decode()
        array = np.frombuffer(row_bytes, dtype=dtype)
        if shape is not None and len(shape) > 0:
            array = array.reshape(shape)
        return array

    def __getitem__(self, index):
        sample = self.samples[index % len(self.samples)]
        if sample["type"] == "metadata":
            return self._load_metadata_sample(sample["row"], Path(sample["base_dir"]))
        if sample["type"] in {"lmdb_sample", "lmdb_wan"}:
            return self._load_lmdb_sample(sample)
        raise AssertionError(f"Unhandled latent dataset sample type: {sample['type']}")

    def _load_metadata_sample(self, row, base_dir):
        inputs = {}
        conditioning = {}
        meta = {}
        height = record_value(row, "target_height", "height")
        width = record_value(row, "target_width", "width")
        if height not in (None, "") and width not in (None, ""):
            meta["target_height"] = int(height)
            meta["target_width"] = int(width)
        for key in ("id", "width", "height", "fps", "frames", "duration", "num_frames"):
            if isinstance(row, dict) and key in row:
                meta[key] = row[key]
        target_num_frames = record_value(row, "target_num_frames", "num_frames")
        if target_num_frames not in (None, ""):
            # Ref2AV caches call this target_num_frames to distinguish the
            # generated duration from every reference video's own geometry.
            # Trainers consume the canonical runtime key num_frames.
            meta["target_num_frames"] = int(target_num_frames)
            meta["num_frames"] = int(target_num_frames)
        for key in ("video", "video_path", "audio", "audio_path", "image", "image_path"):
            value = record_value(row, key)
            path = resolve_data_path(value, base_dir)
            if path is not None:
                meta[key] = str(path)
        prompt = prompt_text(row, self.prompt_column, self.prompt_index)
        if prompt:
            conditioning["prompt"] = prompt

        video_latent_path = _resolve_required_path(record_value(row, "video_latent_path"), base_dir, "video_latent_path")
        if video_latent_path is not None:
            meta["video_latent_path"] = str(video_latent_path)
            if not self.defer_latent_loading:
                video_payload = _video_latent_payload(torch.load(video_latent_path, map_location="cpu", weights_only=True))
                inputs["video_latents"] = video_payload
                if torch.is_tensor(video_payload):
                    inputs["latents"] = video_payload
                elif isinstance(video_payload, dict) and torch.is_tensor(video_payload.get("latents")):
                    inputs["latents"] = video_payload["latents"]

        audio_latent_path = _resolve_required_path(record_value(row, "audio_latent_path"), base_dir, "audio_latent_path")
        if audio_latent_path is not None:
            meta["audio_latent_path"] = str(audio_latent_path)
            if not self.defer_latent_loading:
                inputs["audio_latents"] = torch.load(audio_latent_path, map_location="cpu", weights_only=True)

        condition_path = _resolve_required_path(record_value(row, "condition_path"), base_dir, "condition_path")
        if condition_path is not None:
            condition_item = torch.load(condition_path, map_location="cpu", weights_only=False)
            positive = _condition_payload(condition_item)
            conditioning["positive"] = positive
            meta["condition_path"] = str(condition_path)
            self._add_negative_condition(conditioning, condition_item)
        elif self.negative_condition is not None:
            conditioning["negative"] = self.negative_condition

        row_negative_path = _resolve_required_path(record_value(row, "negative_condition_path"), base_dir, "negative_condition_path")
        if row_negative_path is not None:
            conditioning["negative"] = _condition_payload(torch.load(row_negative_path, map_location="cpu", weights_only=False))
            meta["negative_condition_path"] = str(row_negative_path)

        if not inputs and "positive" not in conditioning:
            raise ValueError(f"Latent metadata row must contain video_latent_path and/or condition_path: {row}")
        return {"inputs": inputs, "conditioning": conditioning, "meta": meta}

    def _load_lmdb_sample(self, record):
        source = self.lmdb_envs[record["env_index"]]
        env = source["env"]
        row_index = record["row_index"]

        if source.get("format") == "sample_pt":
            data_key = f"sample_{row_index:08d}".encode()
            with env.begin() as txn:
                row_bytes = txn.get(data_key)
            if row_bytes is None:
                raise KeyError(f"{data_key!r} not found in LMDB dataset.")
            sample = torch.load(io.BytesIO(row_bytes), map_location="cpu", weights_only=False)
            sample = {
                "inputs": sample.get("inputs", {}),
                "conditioning": sample.get("conditioning", {}),
                "meta": sample.get("meta", {}),
            }
            sample["meta"].setdefault("row_index", row_index)
            sample["meta"].setdefault("lmdb_path", source.get("path"))
            return sample

        latents_shape = source["latents_shape"]
        latents = self._retrieve_lmdb_row(env, "latents", np.float16, row_index, shape=latents_shape[1:])
        if latents.ndim == 4:
            latents = latents[None, ...]
        latent_tchw = torch.tensor(latents, dtype=torch.float32)[-1]
        prompt = self._retrieve_lmdb_row(env, "prompts", str, row_index)
        return {
            "inputs": {"latents": latent_tchw.permute(1, 0, 2, 3).contiguous()},
            "conditioning": {"prompt": prompt},
            "meta": {"row_index": row_index, "lmdb_path": source.get("path")},
        }

    def _add_negative_condition(self, conditioning, item):
        if self.negative_condition is not None:
            conditioning["negative"] = self.negative_condition
        elif isinstance(item, dict) and "negative" in item:
            conditioning["negative"] = _condition_payload({"positive": item["negative"]})
        elif isinstance(item, dict) and "negative_conditions" in item:
            conditioning["negative"] = _condition_payload({"positive": item["negative_conditions"]})

    def __len__(self):
        return len(self.samples) * self.dataset_repeat


MINIMAX_H3_CACHE_TASKS = ("t2av", "i2av", "l2av", "fl2av")
MINIMAX_H3_CACHE_BUCKETS = ("landscape", "portrait")


class MiniMaxH3ReferenceCostSampler(Sampler):
    """Build rank-synchronous, cost-local global batches for Ref2AV.

    Every logical data-parallel rank receives a different row from the same
    global cost bucket.  With an even DP world size, each global batch is split
    evenly between landscape and portrait while keeping reference image count
    and packed-condition cost as local as possible.

    The sampler indexes *physical* metadata rows rather than dataset_repeat
    aliases.  A strict epoch uses every row exactly once, with no padding,
    dropping, or cross-rank duplication.  ``rotating_drop`` forms the largest
    duplicate-free balanced subset and rotates the omitted remainder between
    epochs.  ``configure`` maps completed outer DMD iterations to an absolute
    global-batch offset, making resume continue from the exact next block even
    when a dataloader epoch ended mid-outer.  ``balance_image_counts`` selects
    an equal rotating subset from every image-count/orientation cell; audio
    references may coexist when ``require_image_only`` is disabled.
    """

    is_minimax_h3_ref_cost_sampler = True
    SCHEMA_VERSION = 1
    ROTATING_DROP_SCHEMA_VERSION = 2
    ORIENTATIONS = ("landscape", "portrait")
    REMAINDER_POLICIES = ("strict", "rotating_drop")

    def __init__(
        self,
        dataset,
        *,
        num_replicas=None,
        rank=None,
        seed=0,
        cost_key="packed_sequence_tokens_124",
        require_compute_cost=True,
        require_image_only=True,
        image_counts=None,
        require_all_image_counts=True,
        balance_image_counts=False,
        balance_orientation=True,
        strict_full_epoch=True,
        remainder_policy=None,
    ):
        self.dataset = dataset
        self.num_replicas = get_data_parallel_world_size() if num_replicas is None else int(num_replicas)
        self.rank = get_data_parallel_rank() if rank is None else int(rank)
        if self.num_replicas < 2:
            raise ValueError("MiniMax-H3 Ref cost sampling requires at least two DP ranks.")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}.")
        self.seed = int(seed)
        self.cost_key = str(cost_key).strip()
        if not self.cost_key:
            raise ValueError("Reference cost_key must be non-empty.")
        self.require_compute_cost = bool(require_compute_cost)
        self.require_image_only = bool(require_image_only)
        self.balance_orientation = bool(balance_orientation)
        self.strict_full_epoch = bool(strict_full_epoch)
        if remainder_policy is None:
            if not self.strict_full_epoch:
                raise ValueError("strict_full_epoch=false requires an explicit remainder_policy=rotating_drop.")
            remainder_policy = "strict"
        self.remainder_policy = str(remainder_policy).strip().lower()
        if self.remainder_policy not in self.REMAINDER_POLICIES:
            raise ValueError(f"Unsupported Ref remainder_policy={self.remainder_policy!r}; expected one of {self.REMAINDER_POLICIES}.")
        if self.remainder_policy == "strict" and not self.strict_full_epoch:
            raise ValueError("remainder_policy=strict requires strict_full_epoch=true.")
        if self.remainder_policy == "rotating_drop" and self.strict_full_epoch:
            raise ValueError("remainder_policy=rotating_drop requires strict_full_epoch=false.")
        if not self.balance_orientation:
            raise ValueError("MiniMax-H3 ReferenceCostSampler currently requires balance_orientation=true.")
        if self.num_replicas % 2:
            raise ValueError(f"Balanced Ref cost sampling requires an even data-parallel world size, got {self.num_replicas}.")
        self.half_world_size = self.num_replicas // 2

        configured_counts = None if image_counts is None else tuple(dict.fromkeys(int(value) for value in to_list(image_counts)))
        if configured_counts is not None and any(value < 1 for value in configured_counts):
            raise ValueError(f"reference image_counts must be positive, got {configured_counts}.")
        self.configured_image_counts = configured_counts
        self.require_all_image_counts = bool(require_all_image_counts)
        self.balance_image_counts = bool(balance_image_counts)
        self.records = self._index_records()
        observed_counts = tuple(sorted({record["image_count"] for record in self.records}))
        if self.configured_image_counts is None:
            self.image_counts = observed_counts
        else:
            unexpected = sorted(set(observed_counts) - set(self.configured_image_counts))
            missing = sorted(set(self.configured_image_counts) - set(observed_counts))
            if unexpected:
                raise ValueError(f"Ref cache contains reference_image_count values outside the configured set: {unexpected}.")
            if missing and self.require_all_image_counts:
                raise ValueError(f"Ref cache is missing configured reference_image_count values: {missing}.")
            self.image_counts = tuple(value for value in self.configured_image_counts if value in observed_counts)

        orientation_counts = Counter(record["orientation"] for record in self.records)
        if set(orientation_counts) != set(self.ORIENTATIONS):
            raise ValueError(f"Balanced Ref cost sampling requires landscape and portrait rows; got {dict(orientation_counts)}.")
        cell_counts = Counter((record["image_count"], record["orientation"]) for record in self.records)
        self.rows_per_image_orientation_cell = None
        self.dropped_per_image_orientation_cell = {}
        if self.balance_image_counts:
            if self.remainder_policy != "rotating_drop":
                raise ValueError("balance_image_counts=true requires strict_full_epoch=false and remainder_policy=rotating_drop.")
            self.rows_per_image_orientation_cell = (
                min(cell_counts[(image_count, orientation)] for image_count in self.image_counts for orientation in self.ORIENTATIONS) // self.half_world_size * self.half_world_size
            )
            if self.rows_per_image_orientation_cell < self.half_world_size:
                raise ValueError(
                    "Uniform Ref image-count sampling needs at least half_world_size rows "
                    "in every image-count/orientation cell: "
                    f"cell_counts={dict(sorted(cell_counts.items()))}, "
                    f"half_world_size={self.half_world_size}."
                )
            self.dropped_per_image_orientation_cell = {
                f"{image_count}/{orientation}": (cell_counts[(image_count, orientation)] - self.rows_per_image_orientation_cell)
                for image_count in self.image_counts
                for orientation in self.ORIENTATIONS
            }
            self.rows_per_orientation = self.rows_per_image_orientation_cell * len(self.image_counts)
        elif self.remainder_policy == "strict":
            if orientation_counts["landscape"] != orientation_counts["portrait"]:
                raise ValueError(f"Strict 50/50 Ref cost sampling without dropping or repetition requires equal landscape/portrait counts, got {dict(orientation_counts)}.")
            if len(self.records) % self.num_replicas:
                raise ValueError(f"Ref cache row count must be divisible by the data-parallel world size to use every row exactly once: rows={len(self.records)}, dp_world_size={self.num_replicas}.")
            if orientation_counts["landscape"] % self.half_world_size:
                raise ValueError(
                    "Each orientation count must be divisible by half the DP world size for "
                    "strict 50/50 batches: "
                    f"orientation_rows={orientation_counts['landscape']}, "
                    f"half_world_size={self.half_world_size}."
                )
            self.rows_per_orientation = orientation_counts["landscape"]
        else:
            self.rows_per_orientation = min(orientation_counts[orientation] for orientation in self.ORIENTATIONS) // self.half_world_size * self.half_world_size
            if self.rows_per_orientation < self.half_world_size:
                raise ValueError(
                    f"remainder_policy=rotating_drop needs at least half_world_size rows in each orientation: orientation_counts={dict(orientation_counts)}, half_world_size={self.half_world_size}."
                )

        self.orientation_counts = {orientation: orientation_counts[orientation] for orientation in self.ORIENTATIONS}
        self.dropped_per_orientation = {orientation: orientation_counts[orientation] - self.rows_per_orientation for orientation in self.ORIENTATIONS}
        self.epoch_rows = self.rows_per_orientation * len(self.ORIENTATIONS)
        self.num_global_batches = self.epoch_rows // self.num_replicas
        self.start_iteration = 0
        self.gradient_accumulation_iters = None
        self.fake_update_ratio = None
        self.samples_per_outer_iteration = None
        self.epoch = 0
        self._epoch_batches = OrderedDict()
        self.dataset_fingerprint = self._dataset_fingerprint()
        logger.info(
            "[data] Ref cost sampler rows={} dp_world_size={} global_batches={} "
            "orientation_counts={} epoch_rows={} rows_per_orientation={} "
            "dropped_per_orientation={} remainder_policy={} image_counts={} "
            "balance_image_counts={} rows_per_image_orientation_cell={} "
            "cost_key={} cost_range=[{}, {}] "
            "dataset_fingerprint={}",
            len(self.records),
            self.num_replicas,
            self.num_global_batches,
            dict(orientation_counts),
            self.epoch_rows,
            self.rows_per_orientation,
            self.dropped_per_orientation,
            self.remainder_policy,
            dict(sorted(Counter(record["image_count"] for record in self.records).items())),
            self.balance_image_counts,
            self.rows_per_image_orientation_cell,
            self.cost_key,
            min(record["cost"] for record in self.records),
            max(record["cost"] for record in self.records),
            self.dataset_fingerprint,
        )

    @staticmethod
    def _orientation(row):
        value = str(row.get("target_orientation", row.get("aspect_bucket", ""))).strip().lower()
        if value:
            return value
        height = int(row.get("target_height", row.get("height", 0)) or 0)
        width = int(row.get("target_width", row.get("width", 0)) or 0)
        if width > height:
            return "landscape"
        if height > width:
            return "portrait"
        return ""

    @staticmethod
    def _required_int(row, key, dataset_index):
        value = row.get(key)
        if isinstance(value, bool) or value in (None, ""):
            raise ValueError(f"Ref cost sampler metadata row {dataset_index} requires integer {key}. Rebuild the cache manifest with the current Ref2AV cache builder.")
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Ref cost sampler metadata row {dataset_index} has invalid {key}={value!r}.") from error

    def _index_records(self):
        samples = getattr(self.dataset, "samples", None)
        if not isinstance(samples, list) or not samples:
            raise TypeError("MiniMaxH3ReferenceCostSampler requires a non-empty LatentDataset-style dataset.samples list.")
        records = []
        for dataset_index, sample in enumerate(samples):
            if sample.get("type") != "metadata" or not isinstance(sample.get("row"), dict):
                raise TypeError("MiniMaxH3ReferenceCostSampler supports metadata-backed condition caches only.")
            row = sample["row"]
            image_count = self._required_int(row, "reference_image_count", dataset_index)
            video_count = self._required_int(row, "reference_video_count", dataset_index)
            audio_count = self._required_int(row, "reference_audio_count", dataset_index)
            if self.require_image_only and (video_count or audio_count):
                raise ValueError(f"Fixed-duration Ref cost sampling accepts image-only caches, but row {dataset_index} has image/video/audio={image_count}/{video_count}/{audio_count}.")
            orientation = self._orientation(row)
            if orientation not in self.ORIENTATIONS:
                raise ValueError(f"Ref cost sampler row {dataset_index} has invalid orientation {orientation!r}.")
            raw_cost = row.get(self.cost_key)
            if raw_cost in (None, ""):
                if self.require_compute_cost:
                    raise ValueError(f"Ref cost sampler row {dataset_index} has no {self.cost_key!r}. Rebuild the cache manifest with the current Ref2AV cache builder.")
                cost = image_count
            else:
                try:
                    cost = int(raw_cost)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Ref cost sampler row {dataset_index} has invalid {self.cost_key}={raw_cost!r}.") from error
            if cost < 1:
                raise ValueError(f"Ref cost sampler row {dataset_index} has non-positive cost={cost}.")
            records.append(
                {
                    "dataset_index": dataset_index,
                    "condition_path": str(row.get("condition_path", "")),
                    "image_count": image_count,
                    "orientation": orientation,
                    "cost": cost,
                }
            )
        return records

    def _dataset_fingerprint(self):
        digest = hashlib.sha256()
        for record in self.records:
            digest.update((f"{record['dataset_index']}\0{record['condition_path']}\0{record['image_count']}\0{record['orientation']}\0{record['cost']}\n").encode("utf-8"))
        return digest.hexdigest()

    def configure(self, *, start_iteration, gradient_accumulation_iters, fake_update_ratio):
        self.start_iteration = int(start_iteration)
        self.gradient_accumulation_iters = int(gradient_accumulation_iters)
        self.fake_update_ratio = int(fake_update_ratio)
        if self.start_iteration < 0:
            raise ValueError(f"start_iteration must be non-negative, got {self.start_iteration}.")
        if self.gradient_accumulation_iters < 1:
            raise ValueError("gradient_accumulation_iters must be positive.")
        if self.fake_update_ratio < 1:
            raise ValueError("fake_update_ratio must be positive.")
        self.samples_per_outer_iteration = self.gradient_accumulation_iters * (1 + self.fake_update_ratio)
        self._epoch_batches.clear()
        logger.info(
            "[data] configured Ref cost sampler start_iter={} grad_accum={} fake_update_ratio={} samples_per_outer_rank={} absolute_batch_offset={}",
            self.start_iteration,
            self.gradient_accumulation_iters,
            self.fake_update_ratio,
            self.samples_per_outer_iteration,
            self.start_iteration * self.samples_per_outer_iteration,
        )

    def _ordered_pool(self, records, data_epoch, salt):
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(data_epoch) * 1_000_003 + int(salt))
        order = torch.randperm(len(records), generator=generator).tolist()
        shuffled = [records[index] for index in order]
        # Stable sort keeps equal-cost rows shuffled while producing the
        # tightest possible cost windows for each orientation/count cell.
        shuffled.sort(key=lambda record: record["cost"])
        return shuffled

    def _records_for_epoch(self, data_epoch):
        """Return the exact, duplicate-free physical rows used by one data epoch.

        Strict mode retains the historical one-pass behavior.  Rotating-drop
        mode keeps the largest equal landscape/portrait subset whose per-side
        size is divisible by half the DP world size.  The omitted cyclic
        window advances deterministically every epoch, so remainder rows are
        not permanently starved while every global batch remains 50/50.
        """

        if self.remainder_policy == "strict":
            return self.records

        if self.balance_image_counts:
            selected = []
            rows_per_cell = self.rows_per_image_orientation_cell
            for count_index, image_count in enumerate(self.image_counts):
                for orientation_index, orientation in enumerate(self.ORIENTATIONS):
                    pool = [record for record in self.records if record["image_count"] == image_count and record["orientation"] == orientation]
                    generator = torch.Generator()
                    generator.manual_seed(self.seed + 65_537 * (count_index + 1) + 8_191 * (orientation_index + 1))
                    order = torch.randperm(len(pool), generator=generator).tolist()
                    ordered = [pool[index] for index in order]
                    start = (int(data_epoch) * rows_per_cell) % len(ordered)
                    selected.extend(ordered[(start + offset) % len(ordered)] for offset in range(rows_per_cell))
            expected_rows = rows_per_cell * len(self.image_counts) * len(self.ORIENTATIONS)
            selected_indices = [record["dataset_index"] for record in selected]
            if len(selected_indices) != expected_rows or len(set(selected_indices)) != expected_rows:
                raise RuntimeError("Uniform Ref image-count selection contains duplicates or has the wrong size.")
            return selected

        selected = []
        for orientation_index, orientation in enumerate(self.ORIENTATIONS):
            pool = [record for record in self.records if record["orientation"] == orientation]
            generator = torch.Generator()
            generator.manual_seed(self.seed + 65_537 * (orientation_index + 1))
            order = torch.randperm(len(pool), generator=generator).tolist()
            ordered = [pool[index] for index in order]
            drop_count = self.dropped_per_orientation[orientation]
            if drop_count:
                start = (int(data_epoch) * drop_count) % len(ordered)
                dropped_positions = {(start + offset) % len(ordered) for offset in range(drop_count)}
                ordered = [record for position, record in enumerate(ordered) if position not in dropped_positions]
            if len(ordered) != self.rows_per_orientation:
                raise RuntimeError(f"Ref rotating-drop selected an unexpected orientation size: orientation={orientation} expected={self.rows_per_orientation} got={len(ordered)}.")
            selected.extend(ordered)

        selected_indices = [record["dataset_index"] for record in selected]
        if len(selected_indices) != self.epoch_rows or len(set(selected_indices)) != self.epoch_rows:
            raise RuntimeError("Ref rotating-drop epoch selection contains duplicates or has the wrong size.")
        return selected

    def _build_epoch_batches(self, data_epoch):
        data_epoch = int(data_epoch)
        selected_records = self._records_for_epoch(data_epoch)
        by_cell = {(image_count, orientation): [] for image_count in self.image_counts for orientation in self.ORIENTATIONS}
        for record in selected_records:
            key = (record["image_count"], record["orientation"])
            if key in by_cell:
                by_cell[key].append(record)

        paired_batches = []
        tail = {orientation: [] for orientation in self.ORIENTATIONS}
        for count_index, image_count in enumerate(self.image_counts):
            ordered = {}
            for orientation_index, orientation in enumerate(self.ORIENTATIONS):
                pool = by_cell[(image_count, orientation)]
                if not pool:
                    raise ValueError(f"Ref cost sampler is missing an orientation/image-count cell: {orientation}/{image_count}.")
                ordered[orientation] = self._ordered_pool(
                    pool,
                    data_epoch,
                    10_007 * (count_index + 1) + 101 * orientation_index,
                )
            complete = min(len(ordered[orientation]) // self.half_world_size for orientation in self.ORIENTATIONS)
            for block_index in range(complete):
                start = block_index * self.half_world_size
                end = start + self.half_world_size
                paired_batches.append(ordered["landscape"][start:end] + ordered["portrait"][start:end])
            consumed = complete * self.half_world_size
            for orientation in self.ORIENTATIONS:
                tail[orientation].extend(ordered[orientation][consumed:])

        for orientation in self.ORIENTATIONS:
            tail[orientation].sort(key=lambda record: (record["image_count"], record["cost"]))
            if len(tail[orientation]) % self.half_world_size:
                raise RuntimeError(f"Ref cost sampler could not form a strict full tail block for {orientation}: rows={len(tail[orientation])}, half_world_size={self.half_world_size}.")
        tail_blocks = len(tail["landscape"]) // self.half_world_size
        if len(tail["portrait"]) // self.half_world_size != tail_blocks:
            raise RuntimeError("Ref cost sampler orientation tails have different block counts.")
        for block_index in range(tail_blocks):
            start = block_index * self.half_world_size
            end = start + self.half_world_size
            paired_batches.append(tail["landscape"][start:end] + tail["portrait"][start:end])

        if len(paired_batches) != self.num_global_batches:
            raise RuntimeError(f"Ref cost sampler built an unexpected number of global batches: expected={self.num_global_batches}, got={len(paired_batches)}.")
        generator = torch.Generator()
        generator.manual_seed(self.seed + data_epoch * 1_000_003 + 97_409)
        batch_order = torch.randperm(len(paired_batches), generator=generator).tolist()
        batches = []
        for ordered_index in batch_order:
            batch = paired_batches[ordered_index]
            rank_order = torch.randperm(self.num_replicas, generator=generator).tolist()
            batches.append(tuple(batch[index]["dataset_index"] for index in rank_order))

        flattened = [index for batch in batches for index in batch]
        expected = {record["dataset_index"] for record in selected_records}
        if len(flattened) != len(expected) or set(flattened) != expected:
            raise RuntimeError("Ref cost sampler epoch is not an exact one-pass permutation of its selected physical rows.")
        return tuple(batches)

    def _batches_for_epoch(self, data_epoch):
        data_epoch = int(data_epoch)
        cached = self._epoch_batches.get(data_epoch)
        if cached is not None:
            self._epoch_batches.move_to_end(data_epoch)
            return cached
        batches = self._build_epoch_batches(data_epoch)
        self._epoch_batches[data_epoch] = batches
        while len(self._epoch_batches) > 2:
            self._epoch_batches.popitem(last=False)
        return batches

    def sample_index(self, absolute_global_batch, rank=None):
        absolute_global_batch = int(absolute_global_batch)
        if absolute_global_batch < 0:
            raise ValueError("absolute_global_batch must be non-negative.")
        rank = self.rank if rank is None else int(rank)
        if not 0 <= rank < self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {rank}.")
        data_epoch, batch_index = divmod(absolute_global_batch, self.num_global_batches)
        return self._batches_for_epoch(data_epoch)[batch_index][rank]

    def global_batch_summary(self, absolute_global_batch):
        absolute_global_batch = int(absolute_global_batch)
        data_epoch, batch_index = divmod(absolute_global_batch, self.num_global_batches)
        indices = self._batches_for_epoch(data_epoch)[batch_index]
        records = [self.records[index] for index in indices]
        return {
            "data_epoch": data_epoch,
            "batch_index": batch_index,
            "image_counts": dict(sorted(Counter(r["image_count"] for r in records).items())),
            "orientations": dict(sorted(Counter(r["orientation"] for r in records).items())),
            "cost_min": min(r["cost"] for r in records),
            "cost_max": max(r["cost"] for r in records),
        }

    def checkpoint_metadata(self, *, gradient_accumulation_iters=None, fake_update_ratio=None):
        grad_accum = self.gradient_accumulation_iters if gradient_accumulation_iters is None else int(gradient_accumulation_iters)
        fake_ratio = self.fake_update_ratio if fake_update_ratio is None else int(fake_update_ratio)
        metadata = {
            "route_mode": "ref_cost_bucket",
            "schema_version": self.SCHEMA_VERSION,
            "seed": self.seed,
            "cost_key": self.cost_key,
            "require_compute_cost": self.require_compute_cost,
            "balance_orientation": self.balance_orientation,
            "strict_full_epoch": self.strict_full_epoch,
            "balance_image_counts": self.balance_image_counts,
            "image_counts": list(self.image_counts),
            "dataset_rows": len(self.records),
            "dataset_fingerprint": self.dataset_fingerprint,
            "data_parallel_world_size": self.num_replicas,
            "gradient_accumulation_iters": grad_accum,
            "fake_update_ratio": fake_ratio,
        }
        if self.remainder_policy != "strict":
            metadata.update(
                {
                    "schema_version": self.ROTATING_DROP_SCHEMA_VERSION,
                    "remainder_policy": self.remainder_policy,
                    "epoch_rows": self.epoch_rows,
                    "rows_per_orientation": self.rows_per_orientation,
                    "dropped_per_orientation": dict(self.dropped_per_orientation),
                }
            )
            if self.balance_image_counts:
                metadata.update(
                    {
                        "rows_per_image_orientation_cell": (self.rows_per_image_orientation_cell),
                        "dropped_per_image_orientation_cell": dict(self.dropped_per_image_orientation_cell),
                    }
                )
        return metadata

    def __iter__(self):
        if self.samples_per_outer_iteration is None:
            raise RuntimeError("MiniMaxH3ReferenceCostSampler.configure() must be called by the trainer before iterating the dataloader.")
        absolute_start = self.start_iteration * self.samples_per_outer_iteration + self.epoch * self.num_global_batches
        for local_batch in range(self.num_global_batches):
            yield self.sample_index(absolute_start + local_batch)

    def __len__(self):
        return self.num_global_batches

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def _normalize_minimax_h3_task_bucket_order(
    task_order,
    bucket_order,
    task_bucket_order=None,
):
    """Normalize the H3 routing config without changing the legacy schedule.

    When ``task_bucket_order`` is omitted every task uses ``bucket_order``;
    this is exactly the historical global bucket cycle.  An explicit mapping
    lets tasks expose different bucket sets while keeping the outer task cycle
    uniform.
    """

    normalized_tasks = tuple(dict.fromkeys(str(task).strip().lower() for task in to_list(task_order)))
    normalized_buckets = tuple(dict.fromkeys(str(bucket).strip().lower() for bucket in to_list(bucket_order)))
    if not normalized_tasks or not normalized_buckets:
        raise ValueError("MiniMax-H3 task and bucket orders must be non-empty.")
    unknown_tasks = sorted(set(normalized_tasks) - set(MINIMAX_H3_CACHE_TASKS))
    unknown_buckets = sorted(set(normalized_buckets) - set(MINIMAX_H3_CACHE_BUCKETS))
    if unknown_tasks:
        raise ValueError(f"Unsupported MiniMax-H3 sampler tasks: {unknown_tasks}")
    if unknown_buckets:
        raise ValueError(f"Unsupported MiniMax-H3 sampler buckets: {unknown_buckets}")

    if task_bucket_order is None:
        return (
            normalized_tasks,
            normalized_buckets,
            {task: normalized_buckets for task in normalized_tasks},
        )
    if not isinstance(task_bucket_order, Mapping):
        raise TypeError("MiniMax-H3 task_bucket_order must be a mapping from task to ordered buckets.")

    raw_mapping = {str(task).strip().lower(): value for task, value in task_bucket_order.items()}
    unknown_mapping_tasks = sorted(set(raw_mapping) - set(normalized_tasks))
    missing_mapping_tasks = [task for task in normalized_tasks if task not in raw_mapping]
    if unknown_mapping_tasks:
        raise ValueError(f"MiniMax-H3 task_bucket_order contains tasks outside task_order: {unknown_mapping_tasks}")
    if missing_mapping_tasks:
        raise ValueError(f"MiniMax-H3 task_bucket_order must define every task in task_order; missing={missing_mapping_tasks}")

    normalized_mapping = {}
    for task in normalized_tasks:
        task_buckets = tuple(dict.fromkeys(str(bucket).strip().lower() for bucket in to_list(raw_mapping[task])))
        if not task_buckets:
            raise ValueError(f"MiniMax-H3 task_bucket_order[{task!r}] must be non-empty.")
        unknown = sorted(set(task_buckets) - set(MINIMAX_H3_CACHE_BUCKETS))
        disabled = sorted(set(task_buckets) - set(normalized_buckets))
        if unknown:
            raise ValueError(f"Unsupported MiniMax-H3 sampler buckets for task={task}: {unknown}")
        if disabled:
            raise ValueError(f"MiniMax-H3 task_bucket_order[{task!r}] uses buckets not listed in bucket_order: {disabled}")
        normalized_mapping[task] = task_buckets
    return normalized_tasks, normalized_buckets, normalized_mapping


class MiniMaxH3TaskCycleSampler(Sampler):
    """Distributed task/aspect sampler for mixed MiniMax-H3 DMD training.

    ``homogeneous`` preserves the legacy behavior: a complete outer DMD
    iteration (student plus every fake update) consumes one route.  In
    ``stratified`` mode each local micro-batch still contains one route, while
    the global ``data_parallel_world_size * gradient_accumulation_iters``
    slots of every optimizer update are distributed across all active routes.
    Sequence-parallel peers share one logical data-parallel rank and therefore
    receive exactly the same sample; different data-parallel ranks may receive
    different routes without changing the model/FSDP collective order.

    ``start_iteration`` is the number of completed outer iterations.  Both
    modes derive route and shuffle positions from that absolute value, so an
    outer-boundary resume does not restart routing at T2AV.
    """

    is_minimax_h3_task_cycle_sampler = True
    ROUTE_MODES = frozenset({"homogeneous", "stratified"})

    def __init__(
        self,
        dataset,
        task_order=MINIMAX_H3_CACHE_TASKS,
        bucket_order=MINIMAX_H3_CACHE_BUCKETS,
        task_bucket_order=None,
        num_replicas=None,
        rank=None,
        seed=0,
        drop_last=True,
        route_mode="homogeneous",
        balance_route_loss=False,
        require_all_routes_per_optimizer_step=False,
    ):
        self.dataset = dataset
        (
            self.task_order,
            self.bucket_order,
            self.task_bucket_order,
        ) = _normalize_minimax_h3_task_bucket_order(
            task_order,
            bucket_order,
            task_bucket_order,
        )
        self.active_cells = tuple((task, bucket) for task in self.task_order for bucket in self.task_bucket_order[task])
        bucket_cycle_length = 1
        for task in self.task_order:
            task_bucket_count = len(self.task_bucket_order[task])
            bucket_cycle_length = bucket_cycle_length * task_bucket_count // math.gcd(bucket_cycle_length, task_bucket_count)
        self.schedule_period = len(self.task_order) * bucket_cycle_length
        # Task-major ordering keeps aspect pairs adjacent.  With the common
        # two-bucket setup, every even-sized global optimizer batch is exactly
        # 50/50 landscape/portrait while the task remainder rotates over time.
        # Repeating shorter bucket lists to the LCM retains equal task weight
        # for task-specific bucket configurations.
        self.route_schedule = tuple((task, self.task_bucket_order[task][occurrence % len(self.task_bucket_order[task])]) for task in self.task_order for occurrence in range(bucket_cycle_length))
        if len(self.route_schedule) != self.schedule_period:
            raise AssertionError("MiniMax-H3 route schedule has an invalid period.")
        self.route_frequency = Counter(self.route_schedule)

        self.route_mode = str(route_mode).strip().lower()
        if self.route_mode not in self.ROUTE_MODES:
            raise ValueError(f"MiniMax-H3 route_mode must be one of {sorted(self.ROUTE_MODES)}, got {route_mode!r}.")
        self.balance_route_loss = bool(balance_route_loss)
        self.require_all_routes_per_optimizer_step = bool(require_all_routes_per_optimizer_step)
        if self.route_mode != "stratified" and self.require_all_routes_per_optimizer_step:
            raise ValueError("require_all_routes_per_optimizer_step=true requires route_mode=stratified.")

        self.num_replicas = get_data_parallel_world_size() if num_replicas is None else int(num_replicas)
        self.rank = get_data_parallel_rank() if rank is None else int(rank)
        if self.num_replicas < 1:
            raise ValueError(f"num_replicas must be positive, got {self.num_replicas}.")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}.")
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        if not self.drop_last:
            raise ValueError("MiniMax-H3 task-cycle sampling requires drop_last=true so data-parallel ranks never receive the same repeated condition row.")
        self.epoch = 0
        self.start_iteration = 0
        self.samples_per_outer_iteration = None
        self.gradient_accumulation_iters = None
        self.fake_update_ratio = None
        self.global_optimizer_slots = None
        self.blocks_per_cell = None
        self.blocks_by_cell = None
        self.runs_per_epoch = None
        self._stratified_boundary_guard = 0
        self._permutation_cache = {}
        self._pools = self._build_pools()

    @staticmethod
    def _row_bucket(row):
        bucket = str(row.get("aspect_bucket", row.get("target_orientation", ""))).strip().lower()
        if bucket:
            return bucket
        height = int(row.get("target_height", row.get("height", 0)) or 0)
        width = int(row.get("target_width", row.get("width", 0)) or 0)
        if width > height:
            return "landscape"
        if height > width:
            return "portrait"
        return ""

    def _build_pools(self):
        samples = getattr(self.dataset, "samples", None)
        if not isinstance(samples, list) or not samples:
            raise TypeError("MiniMaxH3TaskCycleSampler requires a non-empty LatentDataset-style dataset.samples list.")
        pools = {key: [] for key in self.active_cells}
        # Use each physical metadata row once. LatentDataset.dataset_repeat is
        # intentionally ignored here: virtual copies would let two DP ranks
        # receive different indices that resolve to the same condition row.
        for dataset_index, sample in enumerate(samples):
            if sample.get("type") != "metadata" or not isinstance(sample.get("row"), dict):
                raise TypeError("MiniMaxH3TaskCycleSampler supports metadata-backed condition caches only.")
            row = sample["row"]
            task = str(row.get("task", "")).strip().lower()
            bucket = self._row_bucket(row)
            key = (task, bucket)
            if key in pools:
                pools[key].append(dataset_index)
        missing = [f"{task}/{bucket}" for (task, bucket), pool in pools.items() if not pool]
        if missing:
            raise RuntimeError(f"MiniMax-H3 task-cycle sampler is missing required task buckets: {missing}")
        return pools

    @property
    def num_cells(self):
        return len(self.active_cells)

    def configure(
        self,
        *,
        start_iteration,
        samples_per_outer_iteration=None,
        gradient_accumulation_iters=None,
        fake_update_ratio=None,
    ):
        start_iteration = int(start_iteration)
        if start_iteration < 0:
            raise ValueError(f"start_iteration must be non-negative, got {start_iteration}.")
        if self.route_mode == "stratified":
            if gradient_accumulation_iters is None or fake_update_ratio is None:
                raise ValueError("MiniMax-H3 stratified sampling requires gradient_accumulation_iters and fake_update_ratio.")
            gradient_accumulation_iters = int(gradient_accumulation_iters)
            fake_update_ratio = int(fake_update_ratio)
            if gradient_accumulation_iters < 1:
                raise ValueError(f"gradient_accumulation_iters must be positive, got {gradient_accumulation_iters}.")
            if fake_update_ratio < 1:
                raise ValueError(f"fake_update_ratio must be positive, got {fake_update_ratio}.")
            derived_samples = gradient_accumulation_iters * (1 + fake_update_ratio)
            if samples_per_outer_iteration is not None and int(samples_per_outer_iteration) != derived_samples:
                raise ValueError(f"samples_per_outer_iteration does not match the stratified DMD schedule: configured={samples_per_outer_iteration}, derived={derived_samples}.")
            samples_per_outer_iteration = derived_samples
            self.gradient_accumulation_iters = gradient_accumulation_iters
            self.fake_update_ratio = fake_update_ratio
            self.global_optimizer_slots = self.num_replicas * gradient_accumulation_iters
            if self.require_all_routes_per_optimizer_step and self.global_optimizer_slots < self.num_cells:
                raise RuntimeError(
                    "MiniMax-H3 stratified sampling cannot cover every active route in one "
                    "optimizer step: data_parallel_world_size * "
                    f"gradient_accumulation_iters={self.global_optimizer_slots} < "
                    f"num_routes={self.num_cells}. Increase gradient accumulation or the "
                    "number of logical data-parallel ranks (WORLD_SIZE / SP_SIZE)."
                )

            phase_count = self.schedule_period // math.gcd(
                self.schedule_period,
                self.global_optimizer_slots,
            )
            phase_counts = [self.optimizer_step_route_counts(role_update_index=phase) for phase in range(phase_count)]
            if self.require_all_routes_per_optimizer_step:
                missing_by_phase = {phase: [f"{task}/{bucket}" for task, bucket in self.active_cells if counts.get((task, bucket), 0) == 0] for phase, counts in enumerate(phase_counts)}
                missing_by_phase = {phase: missing for phase, missing in missing_by_phase.items() if missing}
                if missing_by_phase:
                    raise RuntimeError(f"MiniMax-H3 stratified schedule misses active routes in some optimizer steps: {missing_by_phase}.")
            self._stratified_boundary_guard = max(counts.get(key, 0) for counts in phase_counts for key in self.active_cells)
        else:
            if samples_per_outer_iteration is None:
                if gradient_accumulation_iters is None or fake_update_ratio is None:
                    raise ValueError("MiniMax-H3 homogeneous sampling requires samples_per_outer_iteration.")
                samples_per_outer_iteration = int(gradient_accumulation_iters) * (1 + int(fake_update_ratio))
            samples_per_outer_iteration = int(samples_per_outer_iteration)

        if samples_per_outer_iteration < 1:
            raise ValueError(f"samples_per_outer_iteration must be positive, got {samples_per_outer_iteration}.")
        self.start_iteration = start_iteration
        self.samples_per_outer_iteration = samples_per_outer_iteration
        self._permutation_cache.clear()

        if self.route_mode == "stratified":
            # A route can straddle a shuffle-epoch boundary inside one update.
            # Keeping two update windows per cell lets _cell_permutation make
            # those boundary windows disjoint as well as rank-disjoint.
            min_pool_size = max(1, 2 * self._stratified_boundary_guard)
            too_small = [f"{task}/{bucket}={len(self._pools[(task, bucket)])}" for task, bucket in self.active_cells if len(self._pools[(task, bucket)]) < min_pool_size]
            if too_small:
                raise RuntimeError(f"MiniMax-H3 stratified cells need at least {min_pool_size} rows to keep a cross-epoch optimizer window rank-disjoint; too small: {too_small}")
            self.blocks_by_cell = {key: len(pool) // max(1, self._stratified_boundary_guard) for key, pool in self._pools.items()}
            self.blocks_per_cell = min(self.blocks_by_cell.values())
            global_samples_per_outer = self.num_replicas * self.samples_per_outer_iteration
            self.runs_per_epoch = max(
                1,
                sum(len(pool) for pool in self._pools.values()) // global_samples_per_outer,
            )
            summaries = [self.optimizer_step_summary(role_update_index=phase) for phase in range(self.schedule_period // math.gcd(self.schedule_period, self.global_optimizer_slots))]
            logger.info(
                "[data] configured H3 stratified sampler start_iter={} grad_accum={} "
                "fake_update_ratio={} dp_world_size={} global_optimizer_slots={} "
                "samples_per_outer_rank={} outer_iters_per_epoch={} "
                "balance_route_loss={} schedule={} phase_summaries={}",
                self.start_iteration,
                self.gradient_accumulation_iters,
                self.fake_update_ratio,
                self.num_replicas,
                self.global_optimizer_slots,
                self.samples_per_outer_iteration,
                self.runs_per_epoch,
                self.balance_route_loss,
                [f"{task}/{bucket}" for task, bucket in self.route_schedule],
                summaries,
            )
            return

        global_block_size = samples_per_outer_iteration * self.num_replicas
        self.blocks_by_cell = {key: len(pool) // global_block_size for key, pool in self._pools.items()}
        too_small = [f"{task}/{bucket}={len(self._pools[(task, bucket)])}" for (task, bucket), blocks in self.blocks_by_cell.items() if blocks < 1]
        if too_small:
            raise RuntimeError(f"MiniMax-H3 task-cycle cells need at least num_replicas*samples_per_outer_iteration={global_block_size} rows; too small: {too_small}")
        self.blocks_per_cell = min(self.blocks_by_cell.values())
        # A complete schedule period contains the same number of occurrences
        # for every task even when their bucket lists have different lengths.
        # Keeping dataloader epochs period-aligned makes resume independent of
        # where an iterator boundary happens.
        self.runs_per_epoch = self.blocks_per_cell * self.schedule_period
        logger.info(
            "[data] configured H3 task-cycle sampler start_iter={} samples_per_iter_rank={} dp_world_size={} min_blocks_per_cell={} blocks_by_cell={} outer_iters_per_epoch={} order={}",
            self.start_iteration,
            self.samples_per_outer_iteration,
            self.num_replicas,
            self.blocks_per_cell,
            {f"{task}/{bucket}": blocks for (task, bucket), blocks in self.blocks_by_cell.items()},
            self.runs_per_epoch,
            [f"{task}/{bucket}" for task, bucket in (self.expected_group(index) for index in range(self.schedule_period))],
        )

    def expected_group(self, outer_iteration):
        """Return ``(task, bucket)`` for a zero-based outer iteration."""
        outer_iteration = int(outer_iteration)
        if outer_iteration < 0:
            raise ValueError(f"outer_iteration must be non-negative, got {outer_iteration}.")
        task = self.task_order[outer_iteration % len(self.task_order)]
        task_occurrence = outer_iteration // len(self.task_order)
        task_buckets = self.task_bucket_order[task]
        bucket = task_buckets[task_occurrence % len(task_buckets)]
        return task, bucket

    def _role_update_index(self, outer_iteration, stage, fake_update_index=0):
        outer_iteration = int(outer_iteration)
        if outer_iteration < 0:
            raise ValueError(f"outer_iteration must be non-negative, got {outer_iteration}.")
        stage = str(stage).strip().lower()
        if stage == "student":
            return outer_iteration
        if stage != "fake":
            raise ValueError(f"stage must be student or fake, got {stage!r}.")
        if self.fake_update_ratio is None:
            raise RuntimeError("MiniMaxH3TaskCycleSampler.configure() must be called before routing fake updates.")
        fake_update_index = int(fake_update_index)
        if not 0 <= fake_update_index < self.fake_update_ratio:
            raise ValueError(f"fake_update_index must be in [0, {self.fake_update_ratio}), got {fake_update_index}.")
        return outer_iteration * self.fake_update_ratio + fake_update_index

    def _global_micro_slot(self, micro_step, rank=None):
        if self.gradient_accumulation_iters is None:
            raise RuntimeError("MiniMaxH3TaskCycleSampler.configure() must be called before micro routing.")
        micro_step = int(micro_step)
        if not 0 <= micro_step < self.gradient_accumulation_iters:
            raise ValueError(f"micro_step must be in [0, {self.gradient_accumulation_iters}), got {micro_step}.")
        rank = self.rank if rank is None else int(rank)
        if not 0 <= rank < self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {rank}.")
        # Micro-major layout matches the order in which all ranks enter the
        # same FSDP forward/backward call.
        return micro_step * self.num_replicas + rank

    def expected_micro_group(
        self,
        *,
        outer_iteration,
        stage,
        micro_step,
        fake_update_index=0,
        rank=None,
    ):
        """Return the route assigned to one logical DP micro-batch."""

        if self.route_mode != "stratified":
            return self.expected_group(outer_iteration)
        role_update_index = self._role_update_index(
            outer_iteration,
            stage,
            fake_update_index,
        )
        global_micro_slot = self._global_micro_slot(micro_step, rank=rank)
        role_slot = role_update_index * self.global_optimizer_slots + global_micro_slot
        return self.route_schedule[role_slot % self.schedule_period]

    def optimizer_step_route_counts(self, *, role_update_index):
        """Global per-route counts for one student or fake optimizer update."""

        if self.route_mode != "stratified" or self.global_optimizer_slots is None:
            raise RuntimeError("optimizer_step_route_counts() requires a configured stratified sampler.")
        role_update_index = int(role_update_index)
        if role_update_index < 0:
            raise ValueError(f"role_update_index must be non-negative, got {role_update_index}.")
        start = role_update_index * self.global_optimizer_slots
        return Counter(self.route_schedule[(start + offset) % self.schedule_period] for offset in range(self.global_optimizer_slots))

    def microbatch_loss_scale(
        self,
        *,
        outer_iteration,
        stage,
        micro_step,
        fake_update_index=0,
        rank=None,
    ):
        """Scale a route to the configured mean-of-routes objective.

        FSDP averages over logical DP ranks and the trainer averages over
        gradient-accumulation micro-steps.  If a global update has ``n_r``
        samples from route ``r``, multiplying each of them by
        ``N * target_frequency_r / n_r`` makes its total coefficient exactly
        ``target_frequency_r``.  The mean scale remains one.
        """

        if self.route_mode != "stratified" or not self.balance_route_loss:
            return 1.0
        role_update_index = self._role_update_index(
            outer_iteration,
            stage,
            fake_update_index,
        )
        key = self.expected_micro_group(
            outer_iteration=outer_iteration,
            stage=stage,
            micro_step=micro_step,
            fake_update_index=fake_update_index,
            rank=rank,
        )
        count = self.optimizer_step_route_counts(role_update_index=role_update_index)[key]
        target_frequency = self.route_frequency[key] / self.schedule_period
        return self.global_optimizer_slots * target_frequency / count

    def optimizer_step_summary(self, *, role_update_index):
        counts = self.optimizer_step_route_counts(role_update_index=role_update_index)
        summary = {}
        for key in self.active_cells:
            count = counts.get(key, 0)
            if count == 0:
                continue
            target_frequency = self.route_frequency[key] / self.schedule_period
            scale = self.global_optimizer_slots * target_frequency / count if self.balance_route_loss else 1.0
            summary[f"{key[0]}/{key[1]}"] = {
                "count": count,
                "loss_scale": round(float(scale), 8),
            }
        return summary

    def checkpoint_metadata(
        self,
        *,
        gradient_accumulation_iters=None,
        fake_update_ratio=None,
    ):
        """Static routing state needed to validate deterministic resume."""

        grad_accum = self.gradient_accumulation_iters if gradient_accumulation_iters is None else int(gradient_accumulation_iters)
        fake_ratio = self.fake_update_ratio if fake_update_ratio is None else int(fake_update_ratio)
        return {
            "route_mode": self.route_mode,
            "task_order": list(self.task_order),
            "task_bucket_order": {task: list(self.task_bucket_order[task]) for task in self.task_order},
            "route_schedule": [list(key) for key in self.route_schedule],
            "balance_route_loss": self.balance_route_loss,
            "require_all_routes_per_optimizer_step": (self.require_all_routes_per_optimizer_step),
            "data_parallel_world_size": self.num_replicas,
            "gradient_accumulation_iters": grad_accum,
            "fake_update_ratio": fake_ratio,
            "seed": self.seed,
        }

    def _route_occurrences_before(self, role_slot, key):
        role_slot = int(role_slot)
        cycles, remainder = divmod(role_slot, self.schedule_period)
        return cycles * self.route_frequency[key] + sum(route == key for route in self.route_schedule[:remainder])

    def _stratified_sample_ordinal(
        self,
        *,
        outer_iteration,
        stage,
        micro_step,
        fake_update_index=0,
    ):
        """Return a globally unique per-route ordinal for a chronological slot."""

        stage = str(stage).strip().lower()
        global_micro_slot = self._global_micro_slot(micro_step)
        key = self.expected_micro_group(
            outer_iteration=outer_iteration,
            stage=stage,
            micro_step=micro_step,
            fake_update_index=fake_update_index,
        )
        if stage == "student":
            student_slot = outer_iteration * self.global_optimizer_slots + global_micro_slot
            fake_slots_before = outer_iteration * self.fake_update_ratio * self.global_optimizer_slots
            ordinal = self._route_occurrences_before(student_slot, key)
            ordinal += self._route_occurrences_before(fake_slots_before, key)
        else:
            fake_update = self._role_update_index(
                outer_iteration,
                stage,
                fake_update_index,
            )
            fake_slot = fake_update * self.global_optimizer_slots + global_micro_slot
            student_slots_through_current = (outer_iteration + 1) * self.global_optimizer_slots
            ordinal = self._route_occurrences_before(fake_slot, key)
            ordinal += self._route_occurrences_before(
                student_slots_through_current,
                key,
            )
        return key, ordinal

    def _cell_permutation(self, key, data_epoch):
        cache_key = (key, int(data_epoch), self.route_mode)
        cached = self._permutation_cache.get(cache_key)
        if cached is not None:
            return cached
        task, bucket = key
        cell_index = self.bucket_order.index(bucket) * len(self.task_order) + self.task_order.index(task)
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(data_epoch) * 1_000_003 + cell_index * 9_176)
        pool = self._pools[key]
        order = torch.randperm(len(pool), generator=generator).tolist()
        permutation = [pool[index] for index in order]
        if self.route_mode == "stratified" and data_epoch > 0 and self._stratified_boundary_guard > 0:
            previous = self._cell_permutation(key, data_epoch - 1)
            forbidden = set(previous[-self._stratified_boundary_guard :])
            safe = [index for index in permutation if index not in forbidden]
            deferred = [index for index in permutation if index in forbidden]
            if len(safe) < self._stratified_boundary_guard:
                raise RuntimeError(f"MiniMax-H3 stratified shuffle cannot make its cross-epoch window disjoint for route={key}.")
            permutation = safe + deferred
        self._permutation_cache[cache_key] = permutation
        return permutation

    def _stratified_sample_index(
        self,
        *,
        outer_iteration,
        stage,
        micro_step,
        fake_update_index=0,
    ):
        key, ordinal = self._stratified_sample_ordinal(
            outer_iteration=outer_iteration,
            stage=stage,
            micro_step=micro_step,
            fake_update_index=fake_update_index,
        )
        pool_size = len(self._pools[key])
        data_epoch, position = divmod(ordinal, pool_size)
        return self._cell_permutation(key, data_epoch)[position]

    def __iter__(self):
        if self.runs_per_epoch is None:
            raise RuntimeError("MiniMaxH3TaskCycleSampler.configure() must be called by the trainer before iterating the dataloader.")
        if self.route_mode == "stratified":
            for local_run in range(self.runs_per_epoch):
                outer_iteration = self.start_iteration + self.epoch * self.runs_per_epoch + local_run
                for micro_step in range(self.gradient_accumulation_iters):
                    yield self._stratified_sample_index(
                        outer_iteration=outer_iteration,
                        stage="student",
                        micro_step=micro_step,
                    )
                for fake_update_index in range(self.fake_update_ratio):
                    for micro_step in range(self.gradient_accumulation_iters):
                        yield self._stratified_sample_index(
                            outer_iteration=outer_iteration,
                            stage="fake",
                            micro_step=micro_step,
                            fake_update_index=fake_update_index,
                        )
            return

        global_block_size = self.samples_per_outer_iteration * self.num_replicas
        permutation_cache = {}
        for local_run in range(self.runs_per_epoch):
            outer_iteration = self.start_iteration + self.epoch * self.runs_per_epoch + local_run
            key = self.expected_group(outer_iteration)
            task_occurrence = outer_iteration // len(self.task_order)
            cell_occurrence = task_occurrence // len(self.task_bucket_order[key[0]])
            cell_blocks = self.blocks_by_cell[key]
            data_epoch = cell_occurrence // cell_blocks
            block_index = cell_occurrence % cell_blocks
            cache_key = (key, data_epoch)
            permutation = permutation_cache.get(cache_key)
            if permutation is None:
                permutation = self._cell_permutation(key, data_epoch)
                permutation_cache[cache_key] = permutation

            global_start = block_index * global_block_size
            rank_start = global_start + self.rank * self.samples_per_outer_iteration
            rank_end = rank_start + self.samples_per_outer_iteration
            rank_block = permutation[rank_start:rank_end]
            if len(rank_block) != self.samples_per_outer_iteration:
                raise RuntimeError(f"MiniMax-H3 sampler produced an incomplete rank block for {key}: expected={self.samples_per_outer_iteration}, got={len(rank_block)}.")
            yield from rank_block

    def __len__(self):
        if self.runs_per_epoch is None:
            raise RuntimeError("MiniMaxH3TaskCycleSampler.configure() must be called before its length is queried.")
        return self.runs_per_epoch * self.samples_per_outer_iteration

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def _discover_minimax_h3_cache_metadata(
    roots,
    tasks=MINIMAX_H3_CACHE_TASKS,
    buckets=MINIMAX_H3_CACHE_BUCKETS,
    require_all_tasks=True,
    require_all_buckets=False,
    task_bucket_order=None,
):
    """Discover only canonical H3 condition-cache manifests.

    Supported layouts are ``root/task/metadata.jsonl`` (legacy),
    ``root/task/bucket/metadata.jsonl`` and
    ``root/task/bucket/shard/metadata.jsonl``.  Deliberately avoiding an
    unrestricted recursive glob prevents teacher-video/latent manifests from
    being consumed as prompt-condition datasets.
    """
    (
        selected_tasks,
        selected_buckets,
        selected_task_buckets,
    ) = _normalize_minimax_h3_task_bucket_order(
        tasks,
        buckets,
        task_bucket_order,
    )

    manifests = []
    seen = set()
    locations = []
    for raw_root in to_list(roots):
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"MiniMax-H3 cache root is not a directory: {root}")
        for task in selected_tasks:
            task_dir = root / task
            candidates = [task_dir / "metadata.jsonl"]
            for bucket in selected_task_buckets[task]:
                bucket_dir = task_dir / bucket
                candidates.append(bucket_dir / "metadata.jsonl")
                if bucket_dir.is_dir():
                    candidates.extend(sorted(bucket_dir.glob("*/metadata.jsonl")))
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                manifests.append(resolved)
                locations.append((resolved, task))
    if not manifests:
        raise RuntimeError(f"No MiniMax-H3 condition-cache metadata found below data_path={roots}.")

    task_counts = Counter()
    bucket_counts = Counter()
    variant_counts = Counter()
    joint_counts = Counter()
    manifest_counts = Counter()
    condition_owners = {}
    fingerprint_owners = {}
    for manifest, directory_task in locations:
        directory_bucket = manifest.parent.name if manifest.parent.name in MINIMAX_H3_CACHE_BUCKETS else None
        # Sharded layout: task/bucket/shard/metadata.jsonl.
        if directory_bucket is None and manifest.parent.parent.name in MINIMAX_H3_CACHE_BUCKETS:
            directory_bucket = manifest.parent.parent.name
        for row_number, row in enumerate(read_records(manifest), start=1):
            row_task = str(row.get("task", "")).strip().lower()
            if row_task != directory_task:
                raise ValueError(f"MiniMax-H3 cache task mismatch: directory={directory_task}, row={row_task!r}, manifest={manifest}")
            target_height = int(row.get("target_height", 0))
            target_width = int(row.get("target_width", 0))
            row_bucket = str(row.get("aspect_bucket", row.get("target_orientation", ""))).strip().lower()
            if not row_bucket:
                if target_width > target_height:
                    row_bucket = "landscape"
                elif target_height > target_width:
                    row_bucket = "portrait"
            if row_bucket not in selected_task_buckets[row_task]:
                raise ValueError(f"Invalid or disabled H3 cache bucket {row_bucket!r} for task={row_task}: {manifest}")
            if directory_bucket is not None and row_bucket != directory_bucket:
                raise ValueError(f"MiniMax-H3 cache bucket mismatch: directory={directory_bucket}, row={row_bucket}, manifest={manifest}")
            expected_geometry = (768, 1344) if row_bucket == "landscape" else (1344, 768)
            if (target_height, target_width) != expected_geometry:
                raise ValueError(f"MiniMax-H3 {row_bucket} cache must use {expected_geometry[0]}x{expected_geometry[1]}, got {target_height}x{target_width}: {manifest}")
            condition_value = row.get("condition_path")
            if not isinstance(condition_value, str) or not condition_value.strip():
                raise ValueError(f"MiniMax-H3 condition-cache row has no condition_path: {manifest}")
            condition_path = Path(condition_value).expanduser()
            if not condition_path.is_absolute():
                condition_path = manifest.parent / condition_path
            condition_path = condition_path.resolve()
            owner = f"{manifest}:{row_number}"
            previous_owner = condition_owners.get(condition_path)
            if previous_owner is not None:
                raise ValueError(f"Duplicate MiniMax-H3 condition sample discovered in multiple manifests: condition_path={condition_path}, first={previous_owner}, duplicate={owner}")
            condition_owners[condition_path] = owner
            fingerprint = str(row.get("cache_fingerprint", "")).strip()
            if fingerprint:
                previous_owner = fingerprint_owners.get(fingerprint)
                if previous_owner is not None:
                    raise ValueError(f"Duplicate MiniMax-H3 cache_fingerprint discovered in multiple manifests: fingerprint={fingerprint}, first={previous_owner}, duplicate={owner}")
                fingerprint_owners[fingerprint] = owner
            variant = str(row.get("prompt_variant", "unknown"))
            task_counts[row_task] += 1
            bucket_counts[(row_task, row_bucket)] += 1
            variant_counts[(row_task, variant)] += 1
            joint_counts[(row_task, variant, row_bucket)] += 1
            manifest_counts[str(manifest)] += 1

    if require_all_tasks:
        missing = [task for task in selected_tasks if not task_counts[task]]
        if missing:
            raise RuntimeError(f"MiniMax-H3 cache root is missing required tasks: {missing}")
    if require_all_buckets:
        missing = [f"{task}/{bucket}" for task in selected_tasks for bucket in selected_task_buckets[task] if not bucket_counts[(task, bucket)]]
        if missing:
            raise RuntimeError(f"MiniMax-H3 cache root is missing required task buckets: {missing}")

    logger.info(
        "[data] discovered MiniMax-H3 condition caches manifests={} task_counts={} bucket_counts={} variant_counts={} joint_counts={}",
        len(manifests),
        dict(sorted(task_counts.items())),
        {f"{task}/{bucket}": value for (task, bucket), value in sorted(bucket_counts.items())},
        {f"{task}/{variant}": value for (task, variant), value in sorted(variant_counts.items())},
        {f"{task}/{variant}/{bucket}": value for (task, variant, bucket), value in sorted(joint_counts.items())},
    )
    for manifest in manifests:
        logger.info("[data] H3 cache manifest rows={} path={}", manifest_counts[str(manifest)], manifest)
    return manifests


def _build_dataloader(dataset, data_config, train_or_val):
    dp_world_size = get_data_parallel_world_size()
    sampler = None
    shuffle = data_config.get("shuffle", train_or_val == "train")
    drop_last = data_config.get("drop_last", False)
    if train_or_val == "train" and dp_world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_world_size,
            rank=get_data_parallel_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=data_config.get("batch_size", 1),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=data_config.get("num_workers", 8),
        pin_memory=data_config.get("pin_memory", True),
        drop_last=drop_last if sampler is None else False,
    )


@DATA_REGISTER("minimax_h3_ref_cache_dataset")
def build_minimax_h3_ref_cache_dataset(data_config, train_or_val="train", unconditional_prompt=" ", sample_processor=None):
    """Build Ref2AV caches with synchronized global cost buckets."""

    if int(data_config.get("batch_size", 1)) != 1:
        raise ValueError("MiniMax-H3 Ref2AV packed training requires data.train.batch_size=1.")
    dataset = LatentDataset(
        data_paths=data_config["data_path"],
        dataset_repeat=data_config.get("dataset_repeat", 1),
        max_samples=data_config.get("max_samples"),
        prompt_column=data_config.get("prompt_column", "caption"),
        prompt_index=data_config.get("prompt_index", 0),
        negative_condition_path=data_config.get("negative_condition_path"),
        defer_latent_loading=data_config.get("defer_latent_loading", False),
    )
    if train_or_val != "train":
        return _build_dataloader(dataset, data_config, train_or_val)

    sampler_config = data_config.get("reference_cost_sampler", {})
    if not isinstance(sampler_config, Mapping):
        raise TypeError("data.train.reference_cost_sampler must be a mapping.")
    sampler = MiniMaxH3ReferenceCostSampler(
        dataset,
        num_replicas=get_data_parallel_world_size(),
        rank=get_data_parallel_rank(),
        seed=sampler_config.get("seed", data_config.get("seed", 0)),
        cost_key=sampler_config.get("cost_key", "packed_sequence_tokens_124"),
        require_compute_cost=sampler_config.get("require_compute_cost", True),
        require_image_only=sampler_config.get("require_image_only", True),
        image_counts=sampler_config.get("image_counts"),
        require_all_image_counts=sampler_config.get("require_all_image_counts", True),
        balance_image_counts=sampler_config.get("balance_image_counts", False),
        balance_orientation=sampler_config.get("balance_orientation", True),
        strict_full_epoch=sampler_config.get("strict_full_epoch", True),
        remainder_policy=sampler_config.get("remainder_policy"),
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=data_config.get("num_workers", 0),
        pin_memory=data_config.get("pin_memory", False),
        drop_last=False,
    )


@DATA_REGISTER("minimax_h3_cache_dataset")
def build_minimax_h3_cache_dataset(data_config, train_or_val="train", unconditional_prompt=" ", sample_processor=None):
    if int(data_config.get("batch_size", 1)) != 1:
        raise ValueError("MiniMax-H3 mixed condition-cache training currently requires batch_size=1.")
    metadata_paths = _discover_minimax_h3_cache_metadata(
        roots=data_config["data_path"],
        tasks=data_config.get("tasks", MINIMAX_H3_CACHE_TASKS),
        buckets=data_config.get("buckets", MINIMAX_H3_CACHE_BUCKETS),
        require_all_tasks=bool(data_config.get("require_all_tasks", True)),
        require_all_buckets=bool(data_config.get("require_all_buckets", False)),
        task_bucket_order=data_config.get("task_bucket_order"),
    )
    dataset = LatentDataset(
        data_paths=metadata_paths,
        dataset_repeat=data_config.get("dataset_repeat", 1),
        max_samples=data_config.get("max_samples"),
        prompt_column=data_config.get("prompt_column", "caption"),
        prompt_index=data_config.get("prompt_index", 0),
        negative_condition_path=data_config.get("negative_condition_path"),
        defer_latent_loading=data_config.get("defer_latent_loading", False),
    )
    if train_or_val == "train":
        sampler = MiniMaxH3TaskCycleSampler(
            dataset,
            task_order=data_config.get("task_order", data_config.get("tasks", MINIMAX_H3_CACHE_TASKS)),
            bucket_order=data_config.get("bucket_order", data_config.get("buckets", MINIMAX_H3_CACHE_BUCKETS)),
            task_bucket_order=data_config.get("task_bucket_order"),
            num_replicas=get_data_parallel_world_size(),
            rank=get_data_parallel_rank(),
            seed=data_config.get("seed", 0),
            drop_last=data_config.get("drop_last", True),
            route_mode=data_config.get("route_mode", "homogeneous"),
            balance_route_loss=data_config.get("balance_route_loss", False),
            require_all_routes_per_optimizer_step=data_config.get(
                "require_all_routes_per_optimizer_step",
                False,
            ),
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            num_workers=data_config.get("num_workers", 8),
            pin_memory=data_config.get("pin_memory", True),
            drop_last=False,
        )
    return _build_dataloader(dataset, data_config, train_or_val)
