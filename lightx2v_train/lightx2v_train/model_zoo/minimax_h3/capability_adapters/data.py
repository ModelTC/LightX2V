"""MiniMax-H3 cache metadata and data-parallel batch sampling."""

import hashlib
import json
import random
from collections import defaultdict
from itertools import cycle, islice
from pathlib import Path

from torch.utils.data import Sampler

from lightx2v_train.data.cache_dataset import CacheDataset
from lightx2v_train.model_zoo.native.minimax_h3.packing import FPS

FL2AV_TASKS = ("t2av", "i2av", "fl2av")
_GEOMETRY = ("num_frames", "target_height", "target_width")
_TARGET_PATHS = ("video_latent_path", "audio_latent_path")
_TARGET_INPUTS = ("video_latents", "audio_latents")


def _geometry(record):
    return tuple(record[key] for key in _GEOMETRY)


class MiniMaxH3CacheDataset(CacheDataset):
    def __init__(self, config, *, reference, metadata_paths, **kwargs):
        if kwargs.get("generation_shapes") is not None:
            raise ValueError("MiniMax-H3 cached distillation uses the geometry in each metadata record.")
        default_tasks = ("ref2av",) if reference else FL2AV_TASKS
        self.tasks = tuple(config.get("tasks") or default_tasks)
        if set(self.tasks) - set(default_tasks):
            raise ValueError(f"MiniMax-H3 task filter {self.tasks} must select from {default_tasks}.")
        self.ref_counts = config.get("ref_image_counts") or range(1, 7)
        self.require_targets = config.get("require_targets", False)
        self.group_key = "ref_image_count" if reference else "task"
        paths = [Path(path).expanduser().resolve() for path in metadata_paths]
        super().__init__(metadata_paths=paths, **kwargs)

        self.groups = defaultdict(list)
        self.real_by_geometry = defaultdict(list)
        self.geometries = set()
        fingerprint = hashlib.sha256()
        for index, record in enumerate(self.samples):
            self.groups[record[self.group_key]].append(index)
            geometry = _geometry(record)
            self.geometries.add(geometry)
            if record["has_targets"] or record.get("video_latent_path"):
                self.real_by_geometry[geometry].append(index)
            fingerprint.update(json.dumps(record, sort_keys=True, default=str).encode())
        self.fingerprint = fingerprint.hexdigest()

    def _prepare_record(self, record, metadata_path, record_path):
        task = record["task"]
        if task not in self.tasks:
            return None
        if task == "ref2av":
            count = record["ref_image_count"]
            if count < 1:
                raise ValueError(f"{record_path} needs at least one reference image.")
            if count not in self.ref_counts:
                return None
        video, audio = (record.get(key) for key in _TARGET_PATHS)
        if bool(video) != bool(audio):
            raise ValueError(f"{record_path} must provide both AV latent paths.")
        embedded = record.get("has_targets", False)
        if video and embedded:
            raise ValueError(f"{record_path} must choose embedded or separate AV targets.")
        if self.require_targets and not (video or embedded):
            return None
        if min(_geometry(record)) < 1 or record.get("fps", FPS) != FPS:
            raise ValueError(f"{record_path} requires positive target dimensions and {FPS} fps.")
        prepared = super()._prepare_record(record, metadata_path, record_path)
        if video:
            prepared.update({key: str(metadata_path.parent / record[key]) for key in _TARGET_PATHS})
        return {**record, **prepared, "has_targets": embedded}

    def _prepare_cache(self, cache, path, record):
        condition = cache["conditioning"]
        cached_prompt = condition.get("prompt", cache.get("prompt", record["prompt"]))
        if cached_prompt != record["prompt"]:
            raise ValueError(f"Prompt in {path} differs from the metadata.")
        positive = condition["positive"]
        task = record["task"]
        # Text-only conditions can be reused at other output sizes.
        if positive.get("task", "t2av") != task or (task != "t2av" and (positive["target_num_frames"], positive["target_height"], positive["target_width"]) != _geometry(record)):
            raise ValueError(f"Cached condition task or geometry differs from the metadata: {path}")
        if "cache_fingerprint" in record and record["cache_fingerprint"] != cache.get("cache_fingerprint"):
            raise ValueError(f"Cached condition fingerprint differs from the metadata: {path}")
        if task == "ref2av":
            references = positive["references"]
            if len(references) != record["ref_image_count"] or any(ref["kind"] != "image" for ref in references):
                raise ValueError(f"Cached reference images differ from the metadata: {path}")
        inputs = cache.setdefault("inputs", {})
        if record["has_targets"] and any(inputs.get(key) is None for key in _TARGET_INPUTS):
            raise ValueError(f"{path} must contain the declared paired AV targets.")
        if record.get("video_latent_path"):
            for key in _TARGET_INPUTS:
                inputs.pop(key, None)
        cache.setdefault("meta", {}).update({key: value for key, value in record.items() if key != "training_cache"})

    def build_sampler(self, **kwargs):
        return MiniMaxH3Sampler(self, **kwargs)

    def gan_sample(self, index, geometry):
        indices = self.real_by_geometry.get(geometry)
        if not indices:
            raise ValueError(f"GAN requires paired AV targets for geometry {geometry}.")
        return self[random.Random(index).choice(indices)]

    def validate_gan_targets(self):
        missing = self.geometries - self.real_by_geometry.keys()
        if missing:
            raise ValueError(f"GAN has no paired AV targets for (frames, height, width): {sorted(missing)}")


class MiniMaxH3Sampler(Sampler):
    """Build each global batch deterministically and yield this rank's row."""

    def __init__(self, dataset, *, num_replicas, rank, shuffle, drop_last, seed):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.groups = tuple(sorted(dataset.groups))
        self.mixed_tasks = dataset.group_key == "task" and len(self.groups) > 1
        if self.mixed_tasks:
            quota = num_replicas // len(self.groups)
            if not quota:
                raise ValueError("MiniMax-H3 needs at least one rank per selected FL2AV task.")
            self.num_samples = (max(map(len, dataset.groups.values())) + quota - 1) // quota
        else:
            self.num_samples = (len(dataset) + (0 if drop_last else num_replicas - 1)) // num_replicas
        if not self.num_samples:
            raise ValueError("MiniMax-H3 has too few rows for one global batch with drop_last enabled.")
        self.set_start(0)

    def set_start(self, offset):
        self.start_epoch, self.start_index = divmod(offset, self.num_samples)
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = self.start_epoch + epoch

    def __len__(self):
        return self.num_samples - self._skip()

    def _skip(self):
        return self.start_index if self.epoch == self.start_epoch else 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = self._task_batches(rng) if self.mixed_tasks else self._coverage_batches(rng)
        return (batch[self.rank] for batch in islice(batches, self._skip(), None))

    def _task_batches(self, rng):
        pools = []
        for group in self.groups:
            indices = self.dataset.groups[group]
            pools.append(cycle(rng.sample(indices, len(indices)) if self.shuffle else indices))
        quota, extra = divmod(self.num_replicas, len(pools))
        for step in range(self.num_samples):
            offset = (self.epoch * self.num_samples + step) % len(pools)
            batch = []
            for position, pool in enumerate(pools):
                size = quota + ((position - offset) % len(pools) < extra)
                batch.extend(islice(pool, size))
            if self.shuffle:
                rng.shuffle(batch)
            yield batch

    def _coverage_batches(self, rng):
        # Rotate the omitted tail across epochs before grouping the retained rows.
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed).shuffle(indices)
        remainder = len(indices) % self.num_replicas if self.drop_last else 0
        if remainder:
            start = self.epoch * remainder % len(indices)
            indices = (indices[start:] + indices[:start])[remainder:]
        if self.shuffle:
            rng.shuffle(indices)
        pools = {group: [] for group in self.groups}
        for index in reversed(indices):
            pools[self.dataset.samples[index][self.dataset.group_key]].append(index)
        replay = cycle(indices)
        batches = []
        for step in range(self.num_samples):
            offset = step % len(self.groups)
            order = self.groups[offset:] + self.groups[:offset]
            available = (group for group in order if pools[group])
            batch = [pools[group].pop() for group in islice(available, self.num_replicas)]
            while len(batch) < self.num_replicas:
                pool = max(pools.values(), key=len)
                if not pool:
                    batch.extend(islice(replay, self.num_replicas - len(batch)))
                    break
                batch.append(pool.pop())
            if self.shuffle:
                rng.shuffle(batch)
            batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches
