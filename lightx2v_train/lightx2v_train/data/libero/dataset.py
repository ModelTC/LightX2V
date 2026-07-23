import json
from copy import deepcopy
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler

from lightx2v_train.runtime.distributed import get_data_parallel_rank, get_data_parallel_world_size
from lightx2v_train.utils.registry import DATA_REGISTER

from .processor import FastWAMProcessor
from .robot_video_dataset import RobotVideoDataset


def _default_shape_meta():
    return {
        "images": [
            {"key": "image", "raw_shape": [3, 512, 512], "shape": [3, 224, 224]},
            {"key": "wrist_image", "raw_shape": [3, 512, 512], "shape": [3, 224, 224]},
        ],
        "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
        "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
    }


def _dataset_roots(config):
    values = config.get("dataset_dirs")
    if isinstance(values, (str, Path)):
        values = [values]
    if not values:
        raise ValueError("LIBERO data.dataset_dirs must contain at least one dataset.")
    return [Path(value).expanduser().resolve() for value in values]


def _load_dataset_features(roots):
    result = []
    for root in roots:
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LIBERO dataset metadata does not exist: {info_path}")
        with info_path.open(encoding="utf-8") as handle:
            info = json.load(handle)
        features = info.get("features")
        if not isinstance(features, dict):
            raise ValueError(f"LIBERO dataset metadata has no feature map: {info_path}")
        result.append((root, features))
    return result


def _feature_width(feature_sets, feature_key):
    widths = {}
    for root, features in feature_sets:
        if feature_key not in features:
            raise ValueError(f"LIBERO dataset {root} is missing feature: {feature_key}")
        shape = features[feature_key].get("shape")
        if not isinstance(shape, list) or len(shape) != 1:
            raise ValueError(f"LIBERO feature {feature_key} must have a one-dimensional shape in {root}, got {shape}")
        widths[root] = int(shape[0])
    unique_widths = set(widths.values())
    if len(unique_widths) != 1:
        raise ValueError(f"LIBERO feature {feature_key} has inconsistent widths: {widths}")
    return unique_widths.pop()


def _image_channels(feature_sets, image_key):
    feature_key = f"observation.images.{image_key}"
    channels = {}
    for root, features in feature_sets:
        if feature_key not in features:
            raise ValueError(f"LIBERO dataset {root} is missing camera feature: {feature_key}")
        feature = features[feature_key]
        shape = feature.get("shape")
        if feature.get("dtype") != "video" or not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f"LIBERO camera feature {feature_key} must be a 3D video in {root}, got {feature}")
        channels[root] = int(feature.get("info", {}).get("video.channels", shape[-1]))
    unique_channels = set(channels.values())
    if len(unique_channels) != 1:
        raise ValueError(f"LIBERO camera {image_key} has inconsistent channel counts: {channels}")
    return unique_channels.pop()


def _normalize_vector_meta(configured, group, raw_width):
    items = [{"key": "default"}] if configured is None else configured
    if not isinstance(items, list) or not items:
        raise ValueError(f"shape_meta.{group} must be a non-empty list when provided.")

    normalized = []
    for value in items:
        item = {"key": value} if isinstance(value, str) else deepcopy(value)
        if not isinstance(item, dict) or not item.get("key"):
            raise ValueError(f"Invalid shape_meta.{group} entry: {value}")
        if item["key"] != "default":
            raise ValueError(f"LIBERO currently supports only shape_meta.{group} key 'default', got {item['key']!r}")
        configured_raw_width = int(item.get("raw_shape", raw_width))
        configured_width = int(item.get("shape", configured_raw_width))
        if configured_raw_width != raw_width:
            raise ValueError(f"shape_meta.{group}.default raw width does not match dataset metadata: configured={configured_raw_width}, dataset={raw_width}")
        if configured_width != configured_raw_width:
            raise ValueError(f"shape_meta.{group}.default does not support changing dimensions: raw={configured_raw_width}, processed={configured_width}")
        normalized.append({"key": item["key"], "raw_shape": configured_raw_width, "shape": configured_width})
    return normalized


def _resolve_shape_meta(config):
    roots = _dataset_roots(config)
    feature_sets = _load_dataset_features(roots)
    configured = deepcopy(config.get("shape_meta") or {})
    if not isinstance(configured, dict):
        raise ValueError(f"shape_meta must be a mapping, got {type(configured).__name__}")

    image_size = config.get("image_size", (224, 224))
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError(f"data.image_size must be [height, width], got {image_size}")
    image_size = [int(value) for value in image_size]
    if any(value <= 0 for value in image_size):
        raise ValueError(f"data.image_size values must be positive, got {image_size}")

    configured_images = configured.get("images")
    if configured_images is None:
        configured_images = _default_shape_meta()["images"]
    if not isinstance(configured_images, list) or not configured_images:
        raise ValueError("shape_meta.images must be a non-empty list.")

    images = []
    for value in configured_images:
        item = {"key": value} if isinstance(value, str) else deepcopy(value)
        if not isinstance(item, dict) or not item.get("key"):
            raise ValueError(f"Invalid shape_meta.images entry: {value}")
        channels = _image_channels(feature_sets, item["key"])
        target_shape = [int(dimension) for dimension in item.get("shape", [channels, *image_size])]
        if target_shape != [channels, *image_size]:
            raise ValueError(f"shape_meta.images.{item['key']} shape must match data.image_size and dataset channels: configured={target_shape}, expected={[channels, *image_size]}")
        images.append({"key": item["key"], "shape": target_shape})

    return {
        "images": images,
        "action": _normalize_vector_meta(configured.get("action"), "action", _feature_width(feature_sets, "action")),
        "state": _normalize_vector_meta(
            configured.get("state"),
            "state",
            _feature_width(feature_sets, "observation.state"),
        ),
    }


class DatasetSliceRepeat(torch.utils.data.Dataset):
    def __init__(self, dataset, max_samples=None, dataset_repeat=1):
        self.dataset = dataset
        self.base_len = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
        self.dataset_repeat = max(1, int(dataset_repeat))
        if self.base_len <= 0:
            raise RuntimeError("LIBERO dataset is empty after applying max_samples.")

    def __len__(self):
        return self.base_len * self.dataset_repeat

    def __getitem__(self, index):
        return self.dataset[index % self.base_len]


def _path(value):
    return str(Path(value).expanduser().resolve())


def _build_dataset(config, split):
    shape_meta = _resolve_shape_meta(config)
    num_frames = int(config.get("num_frames", 33))
    processor = FastWAMProcessor(shape_meta, num_frames)
    dataset_dirs = _dataset_roots(config)

    dataset = RobotVideoDataset(
        dataset_dirs=[str(item) for item in dataset_dirs],
        shape_meta=shape_meta,
        processor=processor,
        text_embedding_cache_dir=_path(config["text_embedding_cache_dir"]),
        pretrained_norm_stats=_path(config["pretrained_norm_stats"]),
        num_frames=num_frames,
        context_len=int(config.get("context_len", 128)),
        val_set_proportion=float(config.get("val_set_proportion", 0.0)),
        is_training_set=bool(config.get("is_training_set", split == "train")),
        global_sample_stride=int(config.get("global_sample_stride", 1)),
        action_video_freq_ratio=int(config.get("action_video_freq_ratio", 4)),
        skip_padding_as_possible=bool(config.get("skip_padding_as_possible", False)),
        max_padding_retry=int(config.get("max_padding_retry", 3)),
        video_backend=config.get("video_backend"),
    )
    logger.info("[data] built LIBERO FastWAM {} dataset size={}", split, len(dataset))
    return DatasetSliceRepeat(
        dataset,
        max_samples=config.get("max_samples"),
        dataset_repeat=config.get("dataset_repeat", 1),
    )


@DATA_REGISTER("libero_fastwam_dataset")
def build_libero_fastwam_dataset(data_config, train_or_val="train"):
    dataset = _build_dataset(data_config, train_or_val)
    if data_config.get("return_dataset", False):
        return dataset

    sampler = None
    shuffle = bool(data_config.get("shuffle", train_or_val == "train"))
    world_size = get_data_parallel_world_size()
    if train_or_val == "train" and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=get_data_parallel_rank(),
            shuffle=shuffle,
            drop_last=bool(data_config.get("drop_last", False)),
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 1)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(data_config.get("num_workers", 4)),
        pin_memory=bool(data_config.get("pin_memory", torch.cuda.is_available())),
        drop_last=bool(data_config.get("drop_last", False)),
    )
