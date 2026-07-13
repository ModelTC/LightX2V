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
    shape_meta = deepcopy(config.get("shape_meta") or _default_shape_meta())
    num_frames = int(config.get("num_frames", 33))
    processor = FastWAMProcessor(shape_meta, num_frames)
    dataset_dirs = config.get("dataset_dirs")
    if isinstance(dataset_dirs, (str, Path)):
        dataset_dirs = [dataset_dirs]

    dataset = RobotVideoDataset(
        dataset_dirs=[_path(item) for item in dataset_dirs],
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
