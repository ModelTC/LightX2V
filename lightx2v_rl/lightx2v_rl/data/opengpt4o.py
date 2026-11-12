"""OpenGPT-4o JSONL loading and deterministic, homogeneous task batches."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from lightx2v_rl.data.prepare_opengpt4o import safe_relative_path


def validate_task_cycle(value):
    """Validate the ordered rollout task cycle, including repeated task slots."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("task_cycle must be a nonempty list of t2i/i2i/i2i_single/i2i_multi")
    cycle = tuple(value)
    if any(task not in ("t2i", "i2i", "i2i_single", "i2i_multi") for task in cycle):
        raise ValueError("task_cycle must be a nonempty list of t2i/i2i/i2i_single/i2i_multi")
    return cycle


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

    def get_sampling_task(self, index):
        count = len(self.rows[index]["source_images"])
        return "t2i" if count == 0 else ("i2i_single" if count == 1 else "i2i_multi")

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
            "data_source": f"opengpt4o/{self.get_sampling_task(index)}",
            "reward_model": {"style": "model", "ground_truth": row["prompt"]},
            "extra_info": {
                "index": index,
                "sample_id": row["id"],
                "task_type": row["task_type"],
                "sampling_task": self.get_sampling_task(index),
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
    """Repeat a task cycle with full, homogeneous rollout batches.

    Individual indices are yielded because verl's StatefulDataLoader accepts a
    sampler and a separate batch_size. Each pool is shuffled without replacement
    until exhausted. ``i2i`` includes all editing records; ``i2i_single`` and
    ``i2i_multi`` select exactly one or more than one reference respectively.
    Repeated entries in ``data.task_cycle`` determine task sampling weights.
    The default keeps the original exact 1:1 T2I/I2I alternation.
    Cursor and epoch implement StatefulDataLoader's sampler checkpoint protocol.
    """

    def __init__(self, data_source, data_config=None, *, batch_size=None, seed=None, num_batches=None):
        self.data_source = data_source
        config = data_config or {}
        self.batch_size = int(batch_size or config.get("gen_batch_size") or config.get("train_batch_size", 0))
        if self.batch_size < 1:
            raise ValueError("AlternatingTaskSampler requires a positive batch_size")
        self.seed = int(config.get("seed", 42) if seed is None else seed)
        self.task_cycle = validate_task_cycle(config.get("task_cycle", ("t2i", "i2i")))
        self.by_task = {task: [] for task in self.task_cycle}
        needs_reference_count = bool({"i2i_single", "i2i_multi"}.intersection(self.by_task))
        if needs_reference_count and not callable(getattr(data_source, "get_sampling_task", None)):
            raise ValueError("Reference-specific task_cycle requires a dataset with get_sampling_task(index)")
        for i in range(len(data_source)):
            task = data_source.get_task_type(i)
            if task in self.by_task:
                self.by_task[task].append(i)
            if needs_reference_count:
                sampling_task = data_source.get_sampling_task(i)
                if sampling_task != task and sampling_task in self.by_task:
                    self.by_task[sampling_task].append(i)
        missing = [task for task, indices in self.by_task.items() if not indices]
        if missing:
            raise ValueError(f"AlternatingTaskSampler requires nonempty data for selected tasks: {missing}")
        cycle_size = len(self.task_cycle)
        if num_batches is None:
            frequencies = Counter(self.task_cycle)
            cycles = max(math.ceil(len(indices) / (frequencies[task] * self.batch_size)) for task, indices in self.by_task.items())
            num_batches = cycle_size * cycles
        if num_batches < cycle_size or num_batches % cycle_size:
            raise ValueError("num_batches must be a positive multiple of task_cycle length")
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
        for pool in pools.values():
            rng.shuffle(pool)
        order = []
        for batch in range(self.num_batches):
            task = self.task_cycle[batch % len(self.task_cycle)]
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
            "task_cycle": list(self.task_cycle),
        }

    def load_state_dict(self, state):
        # Checkpoints written before explicit cycles used the default T2I/I2I order.
        if tuple(state.get("task_cycle", ("t2i", "i2i"))) != self.task_cycle:
            raise ValueError("Cannot resume sampler after changing task_cycle")
        for key in ("seed", "batch_size", "length", "dataset_length", "dataset_fingerprint"):
            if state[key] != self.state_dict()[key]:
                raise ValueError(f"Cannot resume sampler after changing {key}")
        if not 0 <= state["cursor"] <= len(self) or state["epoch"] < 0:
            raise ValueError("Invalid sampler checkpoint cursor/epoch")
        self.epoch, self.cursor = int(state["epoch"]), int(state["cursor"])
