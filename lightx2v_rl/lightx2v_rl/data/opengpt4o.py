"""OpenGPT-4o JSONL loading and deterministic, homogeneous T2I/I2I batches."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from lightx2v_rl.data.prepare_opengpt4o import safe_relative_path


class OpenGPT4oDataset(Dataset):
    """Resolve stored image paths relative to each manifest, never the cwd.

    References remain at their native resolution: the official Qwen pipeline
    owns preprocessing, including reference resizing and visual token packing.
    """

    def __init__(self, data_files, tokenizer=None, processor=None, config=None, max_samples=-1):
        del tokenizer, processor
        self.config = config or {}
        paths = [data_files] if isinstance(data_files, (str, Path)) else list(data_files)
        self.rows = []
        seen = set()
        for filename in paths:
            manifest = Path(filename).resolve()
            with manifest.open() as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    task = row.get("task_type")
                    if task not in ("t2i", "i2i") or not row.get("prompt", "").strip():
                        raise ValueError(f"Invalid task/prompt in {manifest.name}:{number}")
                    images = [safe_relative_path(p) for p in row.get("source_images", [])]
                    if (task == "t2i" and images) or (task == "i2i" and not images):
                        raise ValueError(f"Reference count does not match {task}: {manifest.name}:{number}")
                    sample_id = row["id"]
                    if sample_id in seen:
                        raise ValueError(f"Duplicate sample id in manifests: {sample_id}")
                    seen.add(sample_id)
                    row["source_images"] = images
                    row["_manifest_dir"] = manifest.parent
                    self.rows.append(row)
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError("OpenGPT4oDataset received no samples")
        digest = hashlib.sha256()
        for row in self.rows:
            portable = {key: value for key, value in row.items() if key != "_manifest_dir"}
            digest.update(json.dumps(portable, sort_keys=True, ensure_ascii=False).encode())
            digest.update(b"\n")
        self.fingerprint = digest.hexdigest()

    def __len__(self):
        return len(self.rows)

    def get_task_type(self, index):
        return self.rows[index]["task_type"]

    def get_modality(self, index):
        return self.get_task_type(index)

    def __getitem__(self, index):
        row = self.rows[index]
        paths = [str((row["_manifest_dir"] / path).resolve()) for path in row["source_images"]]
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Reference not found; run scripts/assets/prepare_opengpt4o.py first: {path}")
        content = [{"type": "image", "image": path} for path in paths]
        content.append({"type": "text", "text": row["prompt"]})
        raw_prompt = [{"role": "user", "content": content if paths else row["prompt"]}]
        return {
            "dummy_tensor": torch.zeros(1, dtype=torch.uint8),
            "raw_prompt": raw_prompt,
            "prompt": row["prompt"],
            "task_type": row["task_type"],
            "source_images": paths,
            "data_source": f"opengpt4o/{row['task_type']}",
            "reward_model": {"style": "model", "ground_truth": row["prompt"]},
            "extra_info": {
                "index": index,
                "sample_id": row["id"],
                "task_type": row["task_type"],
                "source_images": paths,
                "prompt": row["prompt"],
                "metadata": row.get("metadata", {}),
            },
            "index": index,
            "tools_kwargs": {},
            "interaction_kwargs": {},
        }

    @classmethod
    async def process_vision_info(cls, messages, image_patch_size=None, config=None):
        del image_patch_size, config

        def load_images():
            from PIL import Image

            images = []
            for message in messages:
                content = message.get("content", [])
                if isinstance(content, str):
                    continue
                for block in content:
                    if block.get("type") == "image":
                        value = block["image"]
                        if isinstance(value, Image.Image):
                            images.append(value.copy())
                        else:
                            with Image.open(value) as image:
                                images.append(image.copy())
                    elif block.get("type") in ("video", "audio"):
                        raise ValueError("OpenGPT4oDataset only supports image references")
            return images or None, None

        return await asyncio.to_thread(load_images)


def collate_fn(features):
    """Keep list-valued metadata in 1-D object arrays for DataProto slicing."""
    if not features:
        raise ValueError("Cannot collate an empty image RL batch")
    batch = {}
    for key in features[0]:
        values = [row[key] for row in features]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        else:
            array = np.empty(len(values), dtype=object)
            array[:] = values
            batch[key] = array
    return batch


class AlternatingTaskSampler(Sampler[int]):
    """Alternate full T2I/I2I batches, cycling the smaller task each epoch.

    Individual indices are yielded because verl's StatefulDataLoader accepts a
    sampler and a separate batch_size. Each task is shuffled without replacement
    until exhausted; every epoch has the same number of batches for both tasks.
    Cursor and epoch implement StatefulDataLoader's sampler checkpoint protocol.
    """

    def __init__(self, data_source, data_config=None, *, batch_size=None, seed=None, num_batches=None):
        self.data_source = data_source
        config = data_config or {}
        self.batch_size = int(batch_size or config.get("gen_batch_size") or config.get("train_batch_size", 0))
        if self.batch_size < 1:
            raise ValueError("AlternatingTaskSampler requires a positive batch_size")
        self.seed = int(config.get("seed", 42) if seed is None else seed)
        self.by_task = defaultdict(list)
        for i in range(len(data_source)):
            self.by_task[data_source.get_task_type(i)].append(i)
        if set(self.by_task) != {"t2i", "i2i"}:
            raise ValueError("AlternatingTaskSampler requires both T2I and I2I data")
        if num_batches is None:
            num_batches = 2 * math.ceil(max(map(len, self.by_task.values())) / self.batch_size)
        if num_batches < 2 or num_batches % 2:
            raise ValueError("num_batches must be a positive even number for exact 1:1 alternation")
        self.num_batches = int(num_batches)
        self.epoch = 0
        self.cursor = 0
        self.dataset_fingerprint = getattr(data_source, "fingerprint", None)

    def __len__(self):
        return self.num_batches * self.batch_size

    def _indices(self):
        rng = random.Random(self.seed + self.epoch)
        pools = {task: list(indices) for task, indices in self.by_task.items()}
        offsets = {task: 0 for task in pools}
        for task in ("t2i", "i2i"):
            rng.shuffle(pools[task])
        order = []
        for batch in range(self.num_batches):
            task = ("t2i", "i2i")[batch % 2]
            for _ in range(self.batch_size):
                if offsets[task] == len(pools[task]):
                    rng.shuffle(pools[task])
                    offsets[task] = 0
                order.append(pools[task][offsets[task]])
                offsets[task] += 1
        return order

    def __iter__(self):
        indices = self._indices()
        while self.cursor < len(indices):
            index = indices[self.cursor]
            self.cursor += 1
            yield index
        self.epoch += 1
        self.cursor = 0

    def set_epoch(self, epoch):
        self.epoch, self.cursor = int(epoch), 0

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "length": len(self),
            "dataset_length": len(self.data_source),
            "dataset_fingerprint": self.dataset_fingerprint,
        }

    def load_state_dict(self, state):
        for key in ("seed", "batch_size", "length", "dataset_length", "dataset_fingerprint"):
            if state[key] != self.state_dict()[key]:
                raise ValueError(f"Cannot resume sampler after changing {key}")
        if not 0 <= state["cursor"] <= len(self) or state["epoch"] < 0:
            raise ValueError("Invalid sampler checkpoint cursor/epoch")
        self.epoch, self.cursor = int(state["epoch"]), int(state["cursor"])
