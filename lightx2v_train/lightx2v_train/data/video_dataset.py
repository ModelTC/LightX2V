import io
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler

from lightx2v_train.data.utils import (
    VideoFrameSampler,
    load_video_tensor,
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


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_paths,
        height=480,
        width=832,
        num_frames=81,
        dataset_repeat=1,
        prompt_dropout_rate=0.0,
        video_column="video",
        audio_column="audio",
        image_column="image",
        prompt_column="caption",
        video_root=None,
        audio_root=None,
        image_root=None,
        media_root=None,
        skip_missing=True,
        max_samples=None,
        random_start=False,
        frame_rate=24,
        fix_frame_rate=False,
        decode_retries=3,
    ):
        self.metadata_paths = [_metadata_path(path) for path in to_list(metadata_paths)]
        self.height = int(height)
        self.width = int(width)
        if self.height % 16 != 0 or self.width % 16 != 0:
            raise ValueError(f"Video training height and width must be divisible by 16, got {self.height}x{self.width}.")
        self.dataset_repeat = int(dataset_repeat)
        self.prompt_dropout_rate = float(prompt_dropout_rate)
        self.video_column = video_column
        self.audio_column = audio_column
        self.image_column = image_column
        self.prompt_column = prompt_column
        self.video_roots = [Path(path) for path in to_list(video_root)] + [Path(path) for path in to_list(media_root)]
        self.audio_roots = [Path(path) for path in to_list(audio_root)] + [Path(path) for path in to_list(media_root)]
        self.image_roots = [Path(path) for path in to_list(image_root)] + [Path(path) for path in to_list(media_root)]
        self.skip_missing = bool(skip_missing)
        self.max_samples = max_samples
        self.decode_retries = max(1, int(decode_retries))
        self.frame_sampler = VideoFrameSampler(
            num_frames=num_frames,
            frame_rate=frame_rate,
            fix_frame_rate=fix_frame_rate,
            random_start=random_start,
        )
        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(f"No usable video samples found from data_path={metadata_paths}")
        logger.info("[data] video_dataset samples={} repeat={}", len(self.samples), self.dataset_repeat)

    def _load_samples(self):
        samples = []
        for metadata_path in self.metadata_paths:
            for row in read_records(metadata_path, prompt_column=self.prompt_column):
                video_value = record_value(row, self.video_column, "video_path", "video")
                video_path = resolve_data_path(video_value, metadata_path.parent, self.video_roots, subdirs=("video", "videos"))
                if self.skip_missing and (video_path is None or not video_path.is_file()):
                    continue

                meta = {
                    "video_path": str(video_path or video_value),
                }
                height = record_value(row, "target_height", "height")
                width = record_value(row, "target_width", "width")
                if height not in (None, "") and width not in (None, ""):
                    meta["target_height"] = int(height)
                    meta["target_width"] = int(width)
                for key in ("id", "width", "height", "fps", "frames", "duration", "num_frames"):
                    if isinstance(row, dict) and key in row:
                        meta[key] = row[key]

                audio_value = record_value(row, self.audio_column, "audio_path", "audio")
                audio_path = resolve_data_path(audio_value, metadata_path.parent, self.audio_roots, subdirs=("audio", "audios"))
                if audio_path is not None:
                    meta["audio_path"] = str(audio_path)

                image_value = record_value(row, self.image_column, "image_path", "image")
                image_path = resolve_data_path(image_value, metadata_path.parent, self.image_roots, subdirs=("image", "images"))
                if image_path is not None:
                    meta["image_path"] = str(image_path)

                samples.append({"prompt": prompt_text(row, self.prompt_column), "meta": meta})
                if self.max_samples is not None and len(samples) >= int(self.max_samples):
                    return samples
        return samples

    def _load_video(self, video_path):
        return load_video_tensor(video_path, self.height, self.width, self.frame_sampler)

    def __getitem__(self, index):
        base_index = index % len(self.samples)
        last_error = None
        for retry_id in range(self.decode_retries):
            record = self.samples[(base_index + retry_id) % len(self.samples)]
            meta = dict(record["meta"])
            try:
                prompt = record["prompt"]
                if random.random() < self.prompt_dropout_rate:
                    prompt = " "
                return {
                    "inputs": {"video": self._load_video(meta["video_path"])},
                    "conditioning": {"prompt": prompt},
                    "meta": meta,
                }
            except Exception as error:
                last_error = error
                logger.warning("Failed to load video {}: {}", meta.get("video_path"), error)
        raise last_error

    def __len__(self):
        return len(self.samples) * self.dataset_repeat


class PromptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        prompt_paths,
        dataset_repeat=1,
        max_samples=None,
        prompt_dropout_rate=0.0,
    ):
        self.prompt_paths = [Path(path) for path in to_list(prompt_paths)]
        if not self.prompt_paths:
            raise ValueError("prompt_dataset requires data_path.")
        self.dataset_repeat = int(dataset_repeat)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.prompt_dropout_rate = float(prompt_dropout_rate)
        self.samples = self._load_prompts()
        if not self.samples:
            raise RuntimeError(f"No prompts found from data_path={prompt_paths}")
        logger.info("[data] prompt_dataset samples={} repeat={}", len(self.samples), self.dataset_repeat)

    def _load_prompts(self):
        samples = []
        for path in self.prompt_paths:
            if path.suffix.lower() not in PROMPT_SUFFIXES:
                raise ValueError(f"prompt_dataset only accepts .txt/.list files, got: {path}")
            with path.open("r", encoding="utf-8") as handle:
                for row_index, line in enumerate(handle):
                    prompt = line.strip()
                    if not prompt:
                        continue
                    samples.append({"prompt": prompt, "meta": {"prompt_path": str(path), "row_index": row_index}})
                    if self.max_samples is not None and len(samples) >= self.max_samples:
                        return samples
        return samples

    def __getitem__(self, index):
        sample = self.samples[index % len(self.samples)]
        prompt = sample["prompt"]
        if random.random() < self.prompt_dropout_rate:
            prompt = " "
        return {"inputs": {}, "conditioning": {"prompt": prompt}, "meta": dict(sample["meta"])}

    def __len__(self):
        return len(self.samples) * self.dataset_repeat


class LatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_paths,
        dataset_repeat=1,
        max_samples=None,
        prompt_column="caption",
        prompt_index=0,
        negative_condition_path=None,
    ):
        self.paths = [Path(path) for path in to_list(data_paths)]
        if not self.paths:
            raise ValueError("latent_dataset requires data_path.")
        self.dataset_repeat = int(dataset_repeat)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.prompt_column = prompt_column
        self.prompt_index = int(prompt_index)
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
            video_payload = _video_latent_payload(torch.load(video_latent_path, map_location="cpu", weights_only=True))
            inputs["video_latents"] = video_payload
            if torch.is_tensor(video_payload):
                inputs["latents"] = video_payload
            elif isinstance(video_payload, dict) and torch.is_tensor(video_payload.get("latents")):
                inputs["latents"] = video_payload["latents"]
            meta["video_latent_path"] = str(video_latent_path)

        audio_latent_path = _resolve_required_path(record_value(row, "audio_latent_path"), base_dir, "audio_latent_path")
        if audio_latent_path is not None:
            inputs["audio_latents"] = torch.load(audio_latent_path, map_location="cpu", weights_only=True)
            meta["audio_latent_path"] = str(audio_latent_path)

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


@DATA_REGISTER("video_dataset")
def build_video_dataset(data_config, train_or_val="train"):
    dataset = VideoDataset(
        metadata_paths=data_config["data_path"],
        height=data_config.get("height", 480),
        width=data_config.get("width", 832),
        num_frames=data_config.get("num_frames", 81),
        dataset_repeat=data_config.get("dataset_repeat", 1),
        prompt_dropout_rate=data_config.get("prompt_dropout_rate", 0.0),
        video_column=data_config.get("video_column", "video"),
        audio_column=data_config.get("audio_column", "audio"),
        image_column=data_config.get("image_column", "image"),
        prompt_column=data_config.get("prompt_column", "caption"),
        video_root=data_config.get("video_root"),
        audio_root=data_config.get("audio_root"),
        image_root=data_config.get("image_root"),
        media_root=data_config.get("media_root"),
        skip_missing=data_config.get("skip_missing", True),
        max_samples=data_config.get("max_samples"),
        random_start=data_config.get("random_start", False),
        frame_rate=data_config.get("frame_rate", 24),
        fix_frame_rate=data_config.get("fix_frame_rate", False),
        decode_retries=data_config.get("decode_retries", 3),
    )
    return _build_dataloader(dataset, data_config, train_or_val)


@DATA_REGISTER("prompt_dataset")
def build_prompt_dataset(data_config, train_or_val="train"):
    dataset = PromptDataset(
        prompt_paths=data_config["data_path"],
        dataset_repeat=data_config.get("dataset_repeat", 1),
        max_samples=data_config.get("max_samples"),
        prompt_dropout_rate=data_config.get("prompt_dropout_rate", 0.0),
    )
    return _build_dataloader(dataset, data_config, train_or_val)


@DATA_REGISTER("latent_dataset")
def build_latent_dataset(data_config, train_or_val="train"):
    dataset = LatentDataset(
        data_paths=data_config["data_path"],
        dataset_repeat=data_config.get("dataset_repeat", 1),
        max_samples=data_config.get("max_samples"),
        prompt_column=data_config.get("prompt_column", "caption"),
        prompt_index=data_config.get("prompt_index", 0),
        negative_condition_path=data_config.get("negative_condition_path"),
    )
    return _build_dataloader(dataset, data_config, train_or_val)
