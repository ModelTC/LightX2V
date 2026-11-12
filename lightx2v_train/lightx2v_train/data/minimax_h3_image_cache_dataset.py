"""Cached T2I conditions for H3-Image's online PDMD rollouts."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from lightx2v_train.runtime.distributed import get_data_parallel_rank, get_data_parallel_world_size
from lightx2v_train.utils.registry import DATA_REGISTER


class MiniMaxH3ImageCacheDataset(Dataset):
    uses_cache_dataset = True

    def __init__(self, data_paths):
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        self.samples = []
        for filename in data_paths:
            manifest = Path(filename)
            if manifest.is_dir():
                manifest = manifest / "metadata.jsonl"
            for line in manifest.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["task"] != "t2i" or row["num_frames"] != 1:
                    raise ValueError("H3-Image cache must contain single-frame T2I samples")
                expected = [24, 1, row["target_height"] // 16, row["target_width"] // 16]
                if row["latent_shape"] != expected or row["target_height"] % 32 or row["target_width"] % 32:
                    raise ValueError("H3-Image cache latent geometry does not match its output resolution")
                path = Path(row["condition_path"])
                row["condition_path"] = str(path if path.is_absolute() else (manifest.parent / path).resolve())
                self.samples.append(row)
        if not self.samples:
            raise ValueError("H3-Image cache contains no samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        cache = torch.load(row["condition_path"], map_location="cpu", weights_only=True)
        if cache["cache_schema_version"] != 1 or cache["cache_fingerprint"] != row["cache_fingerprint"] or cache["conditioning"]["prompt"] != row["prompt"]:
            raise ValueError(f"H3-Image manifest and condition cache disagree: {row['condition_path']}")
        positive = cache["conditioning"]["positive"]
        embeds, tags = positive["prompt_embeds"], positive["text_token_tags"]
        if (
            embeds.ndim != 3
            or embeds.shape[0] != 1
            or embeds.shape[-1] != 5120
            or not 0 < embeds.shape[1] <= 4096
            or embeds.dtype != torch.bfloat16
            or tags.shape != (embeds.shape[1],)
            or tags.dtype != torch.long
            or not bool((tags == 1).all())
        ):
            raise ValueError(f"Invalid H3-Image condition tensor shapes or dtype: {row['condition_path']}")
        cache["generation_shape"] = (row["target_height"], row["target_width"])
        cache["meta"]["training_cache_path"] = row["condition_path"]
        return cache


def _single_sample_collate(samples):
    return samples[0]


@DATA_REGISTER("minimax_h3_image_cache_dataset")
def build_minimax_h3_image_cache_dataset(data_config, train_or_val="train", unconditional_prompt=" ", sample_processor=None):
    del unconditional_prompt, sample_processor
    if int(data_config.get("batch_size", 1)) != 1 or float(data_config.get("prompt_dropout_rate", 0)) != 0:
        raise ValueError("H3-Image PDMD uses batch_size=1 and no prompt dropout")
    dataset = MiniMaxH3ImageCacheDataset(data_config["data_path"])
    shuffle = data_config.get("shuffle", train_or_val == "train")
    sampler = None
    if get_data_parallel_world_size() > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_data_parallel_world_size(),
            rank=get_data_parallel_rank(),
            shuffle=shuffle,
            seed=int(data_config.get("seed", 0)),
            drop_last=bool(data_config.get("drop_last", True)),
        )
    return DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", True)),
        collate_fn=_single_sample_collate,
    )
