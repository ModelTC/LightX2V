import json
import random
from pathlib import Path

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
from lightx2v_train.utils.generation_shapes import (
    GenerationShapeSampler,
    apply_generation_shape,
    parse_generation_shapes,
    split_generation_shape_index,
)
from lightx2v_train.utils.registry import DATA_REGISTER

METADATA_SUFFIXES = {".jsonl", ".json", ".csv"}
PROMPT_SUFFIXES = {".txt", ".list"}


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
        preserve_records=False,
        sample_processor=None,
        unconditional_prompt=" ",
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
        self.preserve_records = bool(preserve_records)
        self.sample_processor = sample_processor
        self.unconditional_prompt = unconditional_prompt
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
                if getattr(self.sample_processor, "requires_audio", False) and (audio_path is None or not audio_path.is_file()):
                    if self.skip_missing:
                        continue
                    raise FileNotFoundError(f"Audio path points to a missing file: {audio_path or audio_value}")
                if audio_path is not None:
                    meta["audio_path"] = str(audio_path)

                image_value = record_value(row, self.image_column, "image_path", "image")
                image_path = resolve_data_path(image_value, metadata_path.parent, self.image_roots, subdirs=("image", "images"))
                if image_path is not None:
                    meta["image_path"] = str(image_path)

                prompt = prompt_text(row, self.prompt_column)
                prompt_path_value = record_value(
                    row,
                    "prompt_path",
                    "text_path",
                )
                if prompt_path_value is not None:
                    prompt_path = resolve_data_path(
                        prompt_path_value,
                        metadata_path.parent,
                        subdirs=("prompt", "prompts", "text", "texts"),
                    )
                    if prompt_path is None or not prompt_path.is_file():
                        if self.skip_missing:
                            continue
                        raise FileNotFoundError(f"prompt_path points to a missing file: {prompt_path}")
                    prompt = " ".join(prompt_path.read_text(encoding="utf-8").split())
                    meta["prompt_path"] = str(prompt_path)

                sample = {"prompt": prompt, "meta": meta}
                if self.preserve_records:
                    sample["_original_record"] = dict(row)
                samples.append(sample)
                if self.max_samples is not None and len(samples) >= int(self.max_samples):
                    return samples
        return samples

    def _load_video(self, video_path):
        return load_video_tensor(
            video_path,
            self.height,
            self.width,
            self.frame_sampler,
            return_start_time=True,
        )

    def __getitem__(self, index):
        base_index = index % len(self.samples)
        last_error = None
        for retry_id in range(self.decode_retries):
            sample_index = (base_index + retry_id) % len(self.samples)
            record = self.samples[sample_index]
            meta = dict(record["meta"])
            try:
                prompt = record["prompt"]
                if random.random() < self.prompt_dropout_rate:
                    prompt = self.unconditional_prompt
                video, video_start_time = self._load_video(meta["video_path"])
                meta["video_start_time"] = video_start_time
                sample = {
                    "inputs": {"video": video},
                    "conditioning": {"prompt": prompt},
                    "meta": meta,
                }
                if self.preserve_records:
                    sample["meta"]["dataset_index"] = sample_index
                return sample if self.sample_processor is None else self.sample_processor(sample)
            except Exception as error:
                last_error = error
                logger.warning("Failed to load video {}: {}", meta.get("video_path"), error)
        raise last_error

    def __len__(self):
        return len(self.samples) * self.dataset_repeat

    def cache_source_record(self, index):
        record = self.samples[index]
        if "_original_record" not in record:
            raise RuntimeError("VideoDataset was not configured with preserve_records=true.")
        return dict(record["_original_record"])

    def cache_source_prompt(self, index):
        return self.samples[index]["prompt"]


class PromptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        prompt_paths,
        dataset_repeat=1,
        max_samples=None,
        prompt_dropout_rate=0.0,
        preserve_records=False,
        unconditional_prompt=" ",
    ):
        self.prompt_paths = [Path(path) for path in to_list(prompt_paths)]
        if not self.prompt_paths:
            raise ValueError("prompt_dataset requires data_path.")
        self.dataset_repeat = int(dataset_repeat)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.prompt_dropout_rate = float(prompt_dropout_rate)
        self.preserve_records = bool(preserve_records)
        self.unconditional_prompt = unconditional_prompt
        self.samples = self._load_prompts()
        if not self.samples:
            raise RuntimeError(f"No prompts found from data_path={prompt_paths}")
        logger.info("[data] prompt_dataset samples={} repeat={}", len(self.samples), self.dataset_repeat)

    def _load_prompts(self):
        samples = []
        for path in self.prompt_paths:
            suffix = path.suffix.lower()
            if suffix not in PROMPT_SUFFIXES | {".json", ".jsonl"}:
                raise ValueError(f"prompt_dataset only accepts .txt/.list/.json/.jsonl files, got: {path}")
            for row_index, record in enumerate(read_records(path)):
                prompt = prompt_text(record)
                if not prompt:
                    continue
                negative_prompt = record_value(record, "negative_prompt", default="")
                if isinstance(negative_prompt, (dict, list)):
                    negative_prompt = json.dumps(negative_prompt, ensure_ascii=False, separators=(",", ":"))
                else:
                    negative_prompt = str(negative_prompt).strip()
                target_height = record_value(record, "target_height")
                target_width = record_value(record, "target_width")
                if (target_height is None) != (target_width is None):
                    raise ValueError(f"Prompt metadata must provide target_height and target_width together: {path}:{row_index + 1}")
                if target_height is not None:
                    target_height = int(target_height)
                    target_width = int(target_width)
                    if target_height <= 0 or target_width <= 0:
                        raise ValueError(f"Prompt target dimensions must be positive, got {target_height}x{target_width}: {path}:{row_index + 1}")
                meta = {"prompt_path": str(path), "row_index": row_index}
                if target_height is not None:
                    meta["target_height"] = target_height
                    meta["target_width"] = target_width
                sample = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "meta": meta,
                }
                if self.preserve_records:
                    sample["_original_record"] = dict(record)
                samples.append(sample)
                if self.max_samples is not None and len(samples) >= self.max_samples:
                    return samples
        return samples

    def __getitem__(self, index):
        index, generation_shape = split_generation_shape_index(index)
        sample_index = index % len(self.samples)
        sample = self.samples[sample_index]
        prompt = sample["prompt"]
        if random.random() < self.prompt_dropout_rate:
            prompt = self.unconditional_prompt
        result = {
            "inputs": {},
            "conditioning": {
                "prompt": prompt,
                "negative_prompt": sample["negative_prompt"],
            },
            "meta": dict(sample["meta"]),
        }
        apply_generation_shape(result, generation_shape)
        if self.preserve_records:
            result["meta"]["dataset_index"] = sample_index
        return result

    def __len__(self):
        return len(self.samples) * self.dataset_repeat

    def cache_source_record(self, index):
        sample = self.samples[index]
        if "_original_record" not in sample:
            raise RuntimeError("PromptDataset was not configured with preserve_records=true.")
        return dict(sample["_original_record"])

    def cache_source_prompt(self, index):
        return self.samples[index]["prompt"]


def _build_dataloader(dataset, data_config, train_or_val):
    dp_world_size = get_data_parallel_world_size()
    sampler = None
    distributed_sampling = train_or_val == "train" or data_config.get("distributed_cache_build", False)
    shuffle = data_config.get("shuffle", train_or_val == "train")
    drop_last = data_config.get("drop_last", False)
    generation_shapes = data_config.get("generation_shapes")
    parsed_generation_shapes = parse_generation_shapes(generation_shapes) if generation_shapes is not None else []
    has_multiple_generation_shapes = len(parsed_generation_shapes) > 1
    if train_or_val == "train" and has_multiple_generation_shapes and not data_config.get("distributed_cache_build", False):
        sampler = GenerationShapeSampler(
            dataset,
            num_replicas=dp_world_size,
            rank=get_data_parallel_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
            generation_shapes=generation_shapes,
        )
        shuffle = False
        # GenerationShapeSampler already pads or truncates every distributed step.
        drop_last = False
    elif distributed_sampling and dp_world_size > 1:
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
        batch_size=1,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=data_config.get("num_workers", 8),
        pin_memory=data_config.get("pin_memory", True),
        drop_last=drop_last if sampler is None else False,
    )


@DATA_REGISTER("video_dataset")
def build_video_dataset(data_config, train_or_val="train", sample_processor=None):
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
        preserve_records=data_config.get("preserve_records", False),
        sample_processor=sample_processor,
        unconditional_prompt=getattr(sample_processor, "unconditional_prompt", " "),
    )
    return _build_dataloader(dataset, data_config, train_or_val)


@DATA_REGISTER("prompt_dataset")
def build_prompt_dataset(data_config, train_or_val="train"):
    dataset = PromptDataset(
        prompt_paths=data_config["data_path"],
        dataset_repeat=data_config.get("dataset_repeat", 1),
        max_samples=data_config.get("max_samples"),
        prompt_dropout_rate=data_config.get("prompt_dropout_rate", 0.0),
        preserve_records=data_config.get("preserve_records", False),
    )
    return _build_dataloader(dataset, data_config, train_or_val)
