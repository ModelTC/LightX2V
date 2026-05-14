import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from lightx2v_train.data.utils import center_crop_to_ratio, resize_to_max_side
from lightx2v_train.utils.registry import DATA_REGISTER


class ImageDataset(Dataset):
    def __init__(
        self,
        metadata_paths,
        image_size=1024,
        prompt_dropout_rate=0.0,
        random_ratio=False,
    ):
        self.image_size = image_size
        self.prompt_dropout_rate = prompt_dropout_rate
        self.random_ratio = random_ratio
        self.samples = []
        for path in metadata_paths:
            path = Path(path)
            self.samples.extend(self._load_metadata_samples(path, data_dir=path.parent))
        if not self.samples:
            raise ValueError(f"No valid image samples found in {metadata_paths}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = sample["prompt"]
        if random.random() < self.prompt_dropout_rate:
            prompt = " "

        item = {"prompt": prompt}
        if sample.get("target_image") is not None:
            item["target_image"] = self.load_image(sample["target_image"])
        if sample.get("source_images"):
            item["source_images"] = [self.load_image(p) for p in sample["source_images"]]
        return item

    def _load_metadata_samples(self, metadata_path, data_dir):
        if metadata_path.suffix != ".jsonl":
            raise ValueError(f"Only metadata list files (.jsonl) are supported, not {metadata_path.suffix}: {metadata_path}")
        records = []
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
        return [self._normalize_record(record, data_dir) for record in records]

    def _normalize_record(self, record, data_dir):
        target_image = record.get("target_image")

        prompt = record.get("prompt")
        if prompt is None:
            raise ValueError("Each metadata record must include prompt.")

        source_images = record.get("source_images", [])

        return {
            "target_image": self._resolve_path(target_image, data_dir) if target_image is not None else None,
            "prompt": str(prompt).strip(),
            "source_images": [self._resolve_path(p, data_dir) for p in source_images],
        }

    def _resolve_path(self, path, data_dir):
        path = Path(path)
        if path.is_absolute():
            return path
        return data_dir / path

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if self.random_ratio:
            ratio_name = random.choice(["default", "1:1", "4:3", "16:9"])
            ratios = {"1:1": (1, 1), "4:3": (4, 3), "16:9": (16, 9)}
            if ratio_name in ratios:
                image = center_crop_to_ratio(image, ratios[ratio_name])

        image = resize_to_max_side(image, self.image_size)
        width, height = image.size
        image = image.resize(((width // 32) * 32, (height // 32) * 32))
        return torch.from_numpy(np.asarray(image).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)


@DATA_REGISTER("image_dataset")
def build_image_dataset(data_config_split, train_or_val="train"):
    data_path = data_config_split["data_path"]
    assert isinstance(data_path, list), f"config['data'][{train_or_val!r}]['data_path'] must be a list"

    image_size = data_config_split.get("image_size", 1024)
    prompt_dropout_rate = data_config_split.get("prompt_dropout_rate", 0.0)
    random_ratio = data_config_split.get("random_ratio", False)
    num_workers = data_config_split.get("num_workers", 8)
    shuffle = data_config_split.get("shuffle", train_or_val == "train")

    dataset = ImageDataset(
        metadata_paths=[Path(p) for p in data_path],
        image_size=image_size,
        prompt_dropout_rate=prompt_dropout_rate,
        random_ratio=random_ratio,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
    )
