import csv
import json
import math
import random
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler

from lightx2v_train.runtime.distributed import get_world_size
from lightx2v_train.utils.registry import DATA_REGISTER


def _pil_resampling(name):
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def crop_and_resize(image, target_height, target_width):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    image = image.resize((resized_width, resized_height), _pil_resampling("BILINEAR"))

    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return image.crop((left, top, left + target_width, top + target_height))


def frame_to_tensor(frame):
    array = np.asarray(frame, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


class FrameSampler:
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_rate=24,
        fix_frame_rate=False,
        random_start=False,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_rate = frame_rate
        self.fix_frame_rate = fix_frame_rate
        self.random_start = random_start

    def available_frames(self, reader):
        total_raw_frames = int(reader.count_frames())
        if not self.fix_frame_rate:
            return total_raw_frames

        meta_data = reader.get_meta_data()
        duration = meta_data.get("duration") or total_raw_frames / meta_data["fps"]
        return int(math.floor(duration * self.frame_rate))

    def sample_count(self, reader):
        num_frames = self.num_frames
        total_frames = self.available_frames(reader)
        if total_frames < num_frames:
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return max(1, num_frames)

    def raw_frame_id(self, sequence_id, raw_frame_rate, total_raw_frames):
        if not self.fix_frame_rate:
            return sequence_id

        target_time = sequence_id / self.frame_rate
        frame_id = int(round(target_time * raw_frame_rate))
        return min(frame_id, total_raw_frames - 1)

    def frame_ids(self, reader):
        raw_frame_rate = reader.get_meta_data().get("fps", self.frame_rate)
        total_raw_frames = int(reader.count_frames())
        num_frames = self.sample_count(reader)

        max_start = max(0, self.available_frames(reader) - num_frames)
        start = random.randint(0, max_start) if self.random_start and max_start > 0 else 0
        return [self.raw_frame_id(start + frame_id, raw_frame_rate, total_raw_frames) for frame_id in range(num_frames)]


class WanT2VVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_paths,
        height=480,
        width=832,
        num_frames=81,
        dataset_repeat=1,
        prompt_dropout_rate=0.0,
        video_column="video",
        prompt_column="caption",
        video_root=None,
        skip_missing=True,
        max_samples=None,
        random_start=False,
        frame_rate=24,
        fix_frame_rate=False,
        decode_retries=3,
    ):
        if isinstance(metadata_paths, (str, Path)):
            metadata_paths = [metadata_paths]
        self.metadata_paths = [Path(path) for path in metadata_paths]
        self.height = height
        self.width = width
        if self.height % 16 != 0 or self.width % 16 != 0:
            raise ValueError(f"Wan T2V training height and width must be divisible by 16, got {self.height}x{self.width}.")
        self.dataset_repeat = dataset_repeat
        self.prompt_dropout_rate = prompt_dropout_rate
        self.video_column = video_column
        self.prompt_column = prompt_column
        self.video_root = Path(video_root) if video_root else None
        self.skip_missing = skip_missing
        self.max_samples = max_samples
        self.decode_retries = max(1, decode_retries)
        self.frame_sampler = FrameSampler(
            num_frames=num_frames,
            frame_rate=frame_rate,
            fix_frame_rate=fix_frame_rate,
            random_start=random_start,
        )
        self.samples = self._load_samples()

        if not self.samples:
            raise RuntimeError(f"No usable video samples found from metadata_paths={metadata_paths}")

    def _load_samples(self):
        samples = []
        for metadata_path in self.metadata_paths:
            for row in self._iter_metadata(metadata_path):
                video = row.get(self.video_column, "")
                prompt = row.get(self.prompt_column, "")
                video_path = self._resolve_video_path(metadata_path, video)
                if self.skip_missing and (video_path is None or not video_path.exists()):
                    continue

                samples.append(
                    {
                        "video_path": str(video_path or video),
                        "prompt": str(prompt) if prompt is not None else "",
                    }
                )

                if self.max_samples is not None and len(samples) >= self.max_samples:
                    return samples
        return samples

    def _iter_metadata(self, metadata_path):
        if metadata_path.suffix == ".jsonl":
            with metadata_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
            return

        if metadata_path.suffix == ".json":
            with metadata_path.open("r", encoding="utf-8") as handle:
                for row in json.load(handle):
                    yield row
            return

        csv.field_size_limit(sys.maxsize)
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)

    def _resolve_video_path(self, metadata_path, video):
        if not video or not str(video).strip():
            return None

        video_path = Path(str(video).strip())
        if video_path.is_absolute():
            candidates = [video_path]
        else:
            metadata_dir = metadata_path.parent
            candidates = [
                metadata_dir / video_path,
                metadata_dir / "video" / video_path.name,
            ]
            if self.video_root is not None:
                candidates.extend([self.video_root / video_path, self.video_root / video_path.name])

        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    def _load_video(self, video_path):
        reader = imageio.get_reader(video_path)
        try:
            frames = []
            for frame_id in self.frame_sampler.frame_ids(reader):
                frame = Image.fromarray(reader.get_data(frame_id)).convert("RGB")
                frame = crop_and_resize(frame, self.height, self.width)
                frames.append(frame_to_tensor(frame))
        finally:
            reader.close()

        return torch.stack(frames, dim=1)

    def __getitem__(self, index):
        base_index = index % len(self.samples)
        last_error = None
        for retry_id in range(self.decode_retries):
            sample = self.samples[(base_index + retry_id) % len(self.samples)]
            try:
                prompt = sample["prompt"]
                if random.random() < self.prompt_dropout_rate:
                    prompt = " "
                return {
                    "prompt": prompt,
                    "video": self._load_video(sample["video_path"]),
                    "video_path": sample["video_path"],
                }
            except Exception as error:
                last_error = error
                logger.warning("Failed to load video {}: {}", sample.get("video_path"), error)
        raise last_error

    def __len__(self):
        return len(self.samples) * self.dataset_repeat


class WanT2VCachedDataset(torch.utils.data.Dataset):
    def __init__(self, cache_paths, dataset_repeat=1, max_samples=None):
        if isinstance(cache_paths, (str, Path)):
            cache_paths = [cache_paths]
        self.cache_paths = self._collect_cache_paths(cache_paths)
        if max_samples is not None:
            self.cache_paths = self.cache_paths[:max_samples]
        self.dataset_repeat = dataset_repeat

        if not self.cache_paths:
            raise RuntimeError(f"No .pt cache files found from cache_paths={cache_paths}")

    def _collect_cache_paths(self, cache_paths):
        result = []
        for cache_path in cache_paths:
            path = Path(cache_path)
            if path.is_dir():
                result.extend(sorted(str(item) for item in path.rglob("*.pt")))
            elif path.suffix == ".pt":
                result.append(str(path))
            elif path.suffix in {".txt", ".list"}:
                with path.open("r", encoding="utf-8") as handle:
                    result.extend(line.strip() for line in handle if line.strip())
        return result

    def __getitem__(self, index):
        path = self.cache_paths[index % len(self.cache_paths)]
        item = torch.load(path, map_location="cpu", weights_only=False)
        latent = item["latent"]
        prompt_embed = item["prompt_embed"]
        if latent.ndim == 5 and latent.shape[0] == 1:
            latent = latent[0]
        if prompt_embed.ndim == 3 and prompt_embed.shape[0] == 1:
            prompt_embed = prompt_embed[0]
        return {
            "latent": latent,
            "prompt_embed": prompt_embed,
            "prompt": item.get("prompt", ""),
            "video_path": item.get("video_path", ""),
            "cache_path": path,
        }

    def __len__(self):
        return len(self.cache_paths) * self.dataset_repeat


class PromptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_paths,
        prompt_column="prompt",
        prompt_index=0,
        dataset_repeat=1,
        max_samples=None,
    ):
        if isinstance(metadata_paths, (str, Path)):
            metadata_paths = [metadata_paths]
        self.metadata_paths = [Path(path) for path in metadata_paths]
        self.prompt_column = prompt_column
        self.prompt_index = int(prompt_index)
        self.dataset_repeat = dataset_repeat
        self.max_samples = max_samples
        self.samples = self._load_samples()

        if not self.samples:
            raise RuntimeError(f"No prompts found from metadata_paths={metadata_paths}")

    def _load_samples(self):
        samples = []
        for metadata_path in self.metadata_paths:
            for sample in self._iter_metadata(metadata_path):
                prompt = sample.get("prompt", "")
                if prompt.strip():
                    samples.append(sample)

                if self.max_samples is not None and len(samples) >= self.max_samples:
                    return samples
        return samples

    def _iter_metadata(self, metadata_path):
        if metadata_path.suffix in {".txt", ".list"}:
            with metadata_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    prompt = line.strip()
                    if prompt:
                        yield {"prompt": prompt}
            return

        if metadata_path.suffix == ".jsonl":
            with metadata_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield self._normalize_json_record(json.loads(line))
            return

        if metadata_path.suffix == ".json":
            with metadata_path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
            if isinstance(records, dict):
                records = records.get("prompts", [records])
            for record in records:
                yield self._normalize_json_record(record)
            return

        csv.field_size_limit(sys.maxsize)
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            first_row = next(reader, None)
            if first_row is None:
                return

            header = [column.strip() for column in first_row]
            if self.prompt_column in header:
                column_to_index = {column: index for index, column in enumerate(header)}
                for row in reader:
                    yield self._normalize_csv_row(row, column_to_index)
            else:
                yield self._normalize_csv_row(first_row)
                for row in reader:
                    yield self._normalize_csv_row(row)

    def _normalize_json_record(self, record):
        if isinstance(record, str):
            return {"prompt": record.strip()}
        if not isinstance(record, dict):
            return {"prompt": str(record).strip()}

        prompt = record.get(self.prompt_column, record.get("prompt", ""))
        prompt = "" if prompt is None else str(prompt).strip()
        sample = {"prompt": prompt}
        height = record.get("target_height", record.get("height"))
        width = record.get("target_width", record.get("width"))
        self._maybe_add_target_hw(sample, height, width)
        return sample

    def _normalize_csv_row(self, row, column_to_index=None):
        if column_to_index is not None:
            prompt_index = column_to_index.get(self.prompt_column, column_to_index.get("prompt", 0))
            prompt = self._row_value(row, prompt_index)
            height_index = column_to_index.get("target_height", column_to_index.get("height"))
            width_index = column_to_index.get("target_width", column_to_index.get("width"))
            height = self._row_value(row, height_index)
            width = self._row_value(row, width_index)
        else:
            prompt = self._row_value(row, self.prompt_index)
            height = None
            width = None

        sample = {"prompt": prompt.strip()}
        self._maybe_add_target_hw(sample, height, width)
        return sample

    def _row_value(self, row, index):
        if index is None or index >= len(row):
            return ""
        return str(row[index])

    def _maybe_add_target_hw(self, sample, height, width):
        if height in (None, "") or width in (None, ""):
            return
        sample["target_height"] = int(height)
        sample["target_width"] = int(width)

    def __getitem__(self, index):
        return self.samples[index % len(self.samples)]

    def __len__(self):
        return len(self.samples) * self.dataset_repeat


def _build_dataloader(dataset, data_config, train_or_val):
    world_size = get_world_size()
    sampler = None
    shuffle = data_config.get("shuffle", train_or_val == "train")
    if train_or_val == "train" and world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=data_config.get("num_workers", 8),
        pin_memory=data_config.get("pin_memory", True),
    )


@DATA_REGISTER("wan_t2v_video_dataset")
def build_wan_t2v_video_dataset(data_config, train_or_val="train"):
    dataset = WanT2VVideoDataset(
        metadata_paths=data_config["data_path"],
        height=data_config.get("height", 480),
        width=data_config.get("width", 832),
        num_frames=data_config.get("num_frames", 81),
        dataset_repeat=data_config.get("dataset_repeat", 1),
        prompt_dropout_rate=data_config.get("prompt_dropout_rate", 0.0),
        video_column=data_config.get("video_column", "video"),
        prompt_column=data_config.get("prompt_column", "caption"),
        video_root=data_config.get("video_root"),
        skip_missing=data_config.get("skip_missing", True),
        max_samples=data_config.get("max_samples"),
        random_start=data_config.get("random_start", False),
        frame_rate=data_config.get("frame_rate", 24),
        fix_frame_rate=data_config.get("fix_frame_rate", False),
        decode_retries=data_config.get("decode_retries", 3),
    )
    return _build_dataloader(dataset, data_config, train_or_val)


@DATA_REGISTER("wan_t2v_cached_dataset")
def build_wan_t2v_cached_dataset(data_config, train_or_val="train"):
    cache_paths = data_config.get("cache_path", data_config.get("data_path"))
    dataset = WanT2VCachedDataset(
        cache_paths=cache_paths,
        dataset_repeat=data_config.get("dataset_repeat", 1),
        max_samples=data_config.get("max_samples"),
    )
    return _build_dataloader(dataset, data_config, train_or_val)


@DATA_REGISTER("prompt_dataset")
def build_prompt_dataset(data_config, train_or_val="val"):
    dataset = PromptDataset(
        metadata_paths=data_config["data_path"],
        prompt_column=data_config.get("prompt_column", "prompt"),
        prompt_index=data_config.get("prompt_index", 0),
        dataset_repeat=data_config.get("dataset_repeat", 1),
        max_samples=data_config.get("max_samples"),
    )
    return _build_dataloader(dataset, data_config, train_or_val)
