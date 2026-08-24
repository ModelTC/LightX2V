import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from lightx2v_train.runtime.distributed import get_data_parallel_rank, get_data_parallel_world_size
from lightx2v_train.utils.registry import DATA_REGISTER


class ImageDataset(Dataset):
    """Decode image records and run an injected sample processor."""

    def __init__(
        self,
        metadata_paths,
        prompt_dropout_rate=0.0,
        sample_processor=None,
        preserve_records=False,
        unconditional_prompt=" ",
    ):
        self.prompt_dropout_rate = prompt_dropout_rate
        self.sample_processor = sample_processor
        self.preserve_records = preserve_records
        self.unconditional_prompt = unconditional_prompt
        self.samples = []
        for path in metadata_paths:
            path = Path(path)
            self.samples.extend(self._load_metadata_samples(path, data_dir=path.parent))
        if not self.samples:
            raise ValueError(f"No valid image samples found in {metadata_paths}")
        if self.sample_processor is None:
            raise ValueError("Raw image records require a sample_processor.")

    def __len__(self):
        return len(self.samples)

    def cache_source_record(self, index):
        record = self.samples[index]
        if "_original_record" not in record:
            raise RuntimeError("ImageDataset was not configured with preserve_records=true.")
        return dict(record["_original_record"])

    def cache_source_prompt(self, index):
        return self.samples[index]["prompt"]

    def __getitem__(self, index):
        record = self.samples[index]
        prompt = record["prompt"]
        if random.random() < self.prompt_dropout_rate:
            prompt = self.unconditional_prompt
        sample = self.load_sample(record, prompt=prompt)
        if self.preserve_records:
            sample["meta"]["dataset_index"] = index
        return sample

    def load_sample(self, record, prompt=None):
        inputs = {}
        if record.get("target_image") is not None:
            inputs["target_image"] = self.load_image(record["target_image"])
        if record.get("source_images"):
            inputs["source_images"] = [self.load_image(path) for path in record["source_images"]]

        meta = {}
        if record.get("target_image") is not None:
            meta["target_image_path"] = str(record["target_image"])
        if record.get("source_images"):
            meta["source_image_paths"] = [str(path) for path in record["source_images"]]
        if record.get("target_height") is not None:
            meta["target_height"] = int(record["target_height"])
        if record.get("target_width") is not None:
            meta["target_width"] = int(record["target_width"])
        sample = {
            "inputs": inputs,
            "conditioning": {"prompt": record["prompt"] if prompt is None else prompt},
            "meta": meta,
        }
        return sample if self.sample_processor is None else self.sample_processor(sample)

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
        normalized = {
            "target_image": self._resolve_path(target_image, data_dir) if target_image is not None else None,
            "prompt": str(prompt).strip(),
            "source_images": [self._resolve_path(p, data_dir) for p in source_images],
            "target_height": record.get("target_height"),
            "target_width": record.get("target_width"),
        }
        if self.preserve_records:
            normalized["_original_record"] = dict(record)
        return normalized

    def _resolve_path(self, path, data_dir):
        path = Path(path)
        if path.is_absolute():
            return path
        return data_dir / path

    @staticmethod
    def load_image(image_path):
        with Image.open(image_path) as image:
            return image.convert("RGB").copy()


@DATA_REGISTER("image_dataset")
def build_image_dataset(
    data_config_split,
    train_or_val="train",
    sample_processor=None,
    unconditional_prompt=" ",
):
    data_path = data_config_split["data_path"]
    assert isinstance(data_path, list), f"config['data'][{train_or_val!r}]['data_path'] must be a list"

    prompt_dropout_rate = data_config_split.get("prompt_dropout_rate", 0.0)
    num_workers = data_config_split.get("num_workers", 8)
    shuffle = data_config_split.get("shuffle", train_or_val == "train")

    dataset = ImageDataset(
        metadata_paths=[Path(p) for p in data_path],
        prompt_dropout_rate=prompt_dropout_rate,
        sample_processor=sample_processor,
        preserve_records=data_config_split.get("preserve_records", False),
        unconditional_prompt=unconditional_prompt,
    )
    dp_world_size = get_data_parallel_world_size()
    sampler = DistributedSampler(dataset, num_replicas=dp_world_size, rank=get_data_parallel_rank(), shuffle=shuffle) if dp_world_size > 1 and train_or_val == "train" else None
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = data_config_split.get("persistent_workers", False)
        loader_kwargs["prefetch_factor"] = data_config_split.get("prefetch_factor", 2)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=data_config_split.get("pin_memory", False),
        **loader_kwargs,
    )
