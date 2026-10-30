"""Complete a new H3 cache pass before each complete old-cache pass.

Ordering is global across logical DP ranks. Neither source is downsampled.
Each source boundary is rounded up to a complete optimizer accumulation window
by repeating rows from that same source. Student reuse in official PDMD does
not advance the fresh-data cursor.
"""

import hashlib
import json
import math
from collections import Counter, OrderedDict, deque
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Sampler

from lightx2v_train.data.minimax_h3_cache_dataset import LatentDataset, _discover_minimax_h3_cache_metadata
from lightx2v_train.runtime.distributed import get_data_parallel_rank, get_data_parallel_world_size
from lightx2v_train.utils.registry import DATA_REGISTER


def _positive_int(value, name):
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _condition_identity(sample):
    value = sample["row"].get("condition_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("H3 curriculum rows require condition_path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(sample["base_dir"]) / path
    return str(path.resolve())


class MiniMaxH3CurriculumCacheDataset(Dataset):
    """Concatenate physical cache rows while retaining their source identity."""

    def __init__(self, new_dataset, old_dataset):
        self.sources = {"new": new_dataset, "old": old_dataset}
        self.samples = []
        self.source_ranges = {}
        self.source_local_indices = []
        seen = set()
        digest = hashlib.sha256()
        for name, dataset in self.sources.items():
            if getattr(dataset, "dataset_repeat", 1) != 1:
                raise ValueError("H3 curriculum owns repetition; each source dataset_repeat must be 1")
            physical = getattr(dataset, "samples", None)
            if not isinstance(physical, list) or not physical:
                raise ValueError(f"H3 curriculum requires a nonempty metadata-backed {name} source")
            start = len(self.samples)
            for index, sample in enumerate(physical):
                if sample.get("type") != "metadata" or not isinstance(sample.get("row"), dict):
                    raise TypeError("H3 curriculum accepts metadata-backed cache datasets only")
                identity = _condition_identity(sample)
                if identity in seen:
                    raise ValueError(f"H3 curriculum repeats one condition cache path: {identity}")
                seen.add(identity)
                self.samples.append(sample)
                self.source_local_indices.append((name, index))
                digest.update(json.dumps({"source": name, "condition_path": identity, "row": sample["row"]}, sort_keys=True, ensure_ascii=False, default=str).encode())
                digest.update(b"\n")
            self.source_ranges[name] = (start, len(self.samples))
        self.fingerprint = digest.hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, local_index = self.source_local_indices[index]
        item = self.sources[name][local_index]
        item["meta"] = dict(item.get("meta", {}), curriculum_source=name, curriculum_source_index=local_index)
        return item


class MiniMaxH3CurriculumSampler(Sampler):
    """Deterministic new -> old passes with an absolute global microbatch cursor."""

    is_minimax_h3_curriculum_sampler = True

    def __init__(
        self,
        dataset,
        *,
        kind,
        num_replicas=None,
        rank=None,
        seed=42,
        task_order=("t2av", "i2av", "fl2av"),
        balance_route_loss=None,
        dmd_update_order="student_first",
        official_pdmd=False,
        cost_key="packed_sequence_tokens_124",
        require_image_only=True,
        old_image_counts=None,
    ):
        if kind not in {"fl2av", "ref2av"}:
            raise ValueError("H3 curriculum kind must be fl2av or ref2av")
        if dmd_update_order not in {"student_first", "fake_first"}:
            raise ValueError("H3 curriculum DMD update order must be student_first or fake_first")
        self.dataset, self.kind = dataset, kind
        self.num_replicas = _positive_int(get_data_parallel_world_size() if num_replicas is None else num_replicas, "num_replicas")
        self.rank = get_data_parallel_rank() if rank is None else int(rank)
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("H3 curriculum rank must be within num_replicas")
        self.seed = int(seed)
        self.task_order = tuple(task_order)
        if not self.task_order or len(set(self.task_order)) != len(self.task_order):
            raise ValueError("H3 curriculum task_order must contain distinct tasks")
        self.balance_route_loss = kind == "fl2av" if balance_route_loss is None else bool(balance_route_loss)
        self.dmd_update_order, self.official_pdmd = dmd_update_order, bool(official_pdmd)
        self.cost_key, self.require_image_only = str(cost_key), bool(require_image_only)
        self.old_image_counts = None if old_image_counts is None else tuple(_positive_int(value, "old image count") for value in old_image_counts)
        if self.old_image_counts is not None and (not self.old_image_counts or len(set(self.old_image_counts)) != len(self.old_image_counts)):
            raise ValueError("H3 old Ref image_counts must contain distinct positive counts")
        # Existing trainer/checkpoint integration uses these capability flags.
        self.is_minimax_h3_task_cycle_sampler = kind == "fl2av"
        self.is_minimax_h3_ref_cost_sampler = kind == "ref2av"
        self.start_microbatch = None
        self.gradient_accumulation_iters = self.fake_update_ratio = None
        self.epoch = 0
        self._plans = OrderedDict()
        self._validate_rows()

    def _validate_rows(self):
        for source, (start, stop) in self.dataset.source_ranges.items():
            observed = Counter()
            for index in range(start, stop):
                row = self.dataset.samples[index]["row"]
                task = str(row.get("task", "")).lower()
                height, width = int(row.get("target_height", 0)), int(row.get("target_width", 0))
                frames = int(row.get("target_num_frames", row.get("num_frames", 0)))
                if frames != 124 or height <= 0 or width <= 0 or height % 32 or width % 32:
                    raise ValueError(f"H3 curriculum requires cached 124-frame, positive 32-aligned native geometry: {source} index {index}")
                if self.kind == "fl2av":
                    expected = {"t2av"} if source == "new" else set(self.task_order)
                    if task not in expected:
                        raise ValueError(f"H3 {source} source has unsupported task={task!r}; expected {sorted(expected)}")
                    observed[task] += 1
                else:
                    if task not in {"ref2av", "ref2va"}:
                        raise ValueError(f"H3 Ref curriculum source requires task=ref2av: {source} index {index}")
                    images = int(row.get("reference_image_count", row.get("ref_image_count", 0)))
                    videos, audios = int(row.get("reference_video_count", 0)), int(row.get("reference_audio_count", 0))
                    if images < 0 or videos < 0 or audios < 0 or images + videos < 1:
                        raise ValueError(f"H3 Ref curriculum expects an image/video reference: {source} index {index}")
                    if source == "old" and self.require_image_only and (images < 1 or videos or audios):
                        raise ValueError(f"H3 old Ref curriculum expects image-only references: index {index}")
                    if source == "old" and self.old_image_counts is not None and images not in self.old_image_counts:
                        raise ValueError(f"H3 old Ref image count {images} is outside configured image_counts")
                    cost = row.get(self.cost_key)
                    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
                        raise ValueError(f"H3 Ref curriculum requires finite positive {self.cost_key}: {source} index {index}")
            if self.kind == "fl2av" and source == "old" and set(observed) != set(self.task_order):
                raise ValueError(f"H3 old curriculum is missing task pools: {sorted(set(self.task_order) - observed.keys())}")

    def _configure(self, *, start_microbatch, gradient_accumulation_iters, fake_update_ratio, official):
        if isinstance(start_microbatch, bool) or int(start_microbatch) != start_microbatch or start_microbatch < 0:
            raise ValueError("H3 curriculum start_microbatch must be a non-negative integer")
        self.gradient_accumulation_iters = _positive_int(gradient_accumulation_iters, "gradient_accumulation_iters")
        self.fake_update_ratio = _positive_int(fake_update_ratio, "fake_update_ratio")
        self.start_microbatch = int(start_microbatch)
        self.official_pdmd = bool(official)
        self.epoch = 0
        self._plans.clear()
        logger.info(
            "[data] H3 curriculum kind={} order=new->old repeat=every_round source_rows={} padding=same_source_to_optimizer_boundary grad_accum={} dp_world={} fresh_microbatch_offset={}",
            self.kind,
            self.source_counts,
            self.gradient_accumulation_iters,
            self.num_replicas,
            self.start_microbatch,
        )

    def configure(self, *, start_iteration, gradient_accumulation_iters, fake_update_ratio):
        if isinstance(start_iteration, bool) or int(start_iteration) != start_iteration or start_iteration < 0:
            raise ValueError("H3 curriculum start_iteration must be a non-negative integer")
        accum = _positive_int(gradient_accumulation_iters, "gradient_accumulation_iters")
        ratio = _positive_int(fake_update_ratio, "fake_update_ratio")
        self._configure(start_microbatch=int(start_iteration) * accum * (1 + ratio), gradient_accumulation_iters=accum, fake_update_ratio=ratio, official=False)

    def configure_from_microbatch_offset(
        self, *, start_microbatch, gradient_accumulation_iters=1, fake_update_ratio=5, data_consumption_mode="optimizer_update_critic_only", student_data_source="reuse_preceding_critic_trajectory"
    ):
        if data_consumption_mode != "optimizer_update_critic_only" or student_data_source != "reuse_preceding_critic_trajectory":
            raise ValueError("H3 official PDMD curriculum requires critic-only data and preceding-critic student reuse")
        self._configure(start_microbatch=start_microbatch, gradient_accumulation_iters=gradient_accumulation_iters, fake_update_ratio=fake_update_ratio, official=True)

    @property
    def source_counts(self):
        return {source: stop - start for source, (start, stop) in self.dataset.source_ranges.items()}

    def _shuffle(self, values, round_index, salt):
        generator = torch.Generator().manual_seed(self.seed + round_index * 1_000_003 + salt)
        order = torch.randperm(len(values), generator=generator).tolist()
        return [values[index] for index in order]

    @staticmethod
    def _pad(values, multiple):
        needed = (-len(values)) % multiple
        return values + [values[index % len(values)] for index in range(needed)]

    def _source_order(self, source, round_index):
        start, stop = self.dataset.source_ranges[source]
        indices = list(range(start, stop))
        if self.kind == "fl2av":
            if source == "new":
                return self._shuffle(indices, round_index, 17)
            queues = {
                task: deque(self._shuffle([index for index in indices if self.dataset.samples[index]["row"]["task"] == task], round_index, 101 + task_index))
                for task_index, task in enumerate(self.task_order)
            }
            ordered = []
            task_index = round_index % len(self.task_order)
            while any(queues.values()):
                task = self.task_order[task_index % len(self.task_order)]
                task_index += 1
                if queues[task]:
                    ordered.append(queues[task].popleft())
            return ordered
        by_count = {}
        for index in indices:
            row = self.dataset.samples[index]["row"]
            by_count.setdefault(int(row.get("reference_image_count", row.get("ref_image_count", 0))), []).append(index)
        batches = []
        for image_count, pool in sorted(by_count.items()):
            pool = self._shuffle(pool, round_index, (19 if source == "new" else 211) + image_count)
            # Stable sorting randomizes tied costs while keeping each DP batch local.
            pool.sort(key=lambda index: self.dataset.samples[index]["row"][self.cost_key])
            pool = self._pad(pool, self.num_replicas)
            batches.extend(pool[index : index + self.num_replicas] for index in range(0, len(pool), self.num_replicas))
        batches = self._shuffle(batches, round_index, 31 if source == "new" else 307)
        return [index for batch in batches for index in batch]

    def _plan(self, round_index):
        if self.gradient_accumulation_iters is None:
            raise RuntimeError("Configure the H3 curriculum sampler before iteration")
        if round_index not in self._plans:
            phases, offsets = {}, {}
            ordered = []
            for source in ("new", "old"):
                values = self._pad(self._source_order(source, round_index), self.num_replicas * self.gradient_accumulation_iters)
                offsets[source] = (len(ordered) // self.num_replicas, (len(ordered) + len(values)) // self.num_replicas)
                phases[source] = values
                ordered.extend(values)
            self._plans[round_index] = (ordered, offsets, {key: len(value) - self.source_counts[key] for key, value in phases.items()})
            while len(self._plans) > 2:
                self._plans.popitem(last=False)
        self._plans.move_to_end(round_index)
        return self._plans[round_index]

    @property
    def microbatches_per_round(self):
        return len(self._plan(0)[0]) // self.num_replicas

    def sample_index(self, absolute_microbatch, rank=None):
        if isinstance(absolute_microbatch, bool) or int(absolute_microbatch) != absolute_microbatch or absolute_microbatch < 0:
            raise ValueError("H3 curriculum absolute_microbatch must be a non-negative integer")
        rank = self.rank if rank is None else int(rank)
        if not 0 <= rank < self.num_replicas:
            raise ValueError("H3 curriculum rank is outside the DP world")
        round_index, within_round = divmod(int(absolute_microbatch), self.microbatches_per_round)
        return self._plan(round_index)[0][within_round * self.num_replicas + rank]

    def phase_at(self, absolute_microbatch):
        round_index, within_round = divmod(int(absolute_microbatch), self.microbatches_per_round)
        _, offsets, _ = self._plan(round_index)
        return "new" if within_round < offsets["new"][1] else "old"

    def cursor_microbatch_loss_scale(self, absolute_microbatch):
        if not self.balance_route_loss or self.kind != "fl2av" or self.phase_at(absolute_microbatch) == "new":
            return 1.0
        start = (int(absolute_microbatch) // self.gradient_accumulation_iters) * self.gradient_accumulation_iters
        counts = Counter(self.dataset.samples[self.sample_index(start + micro, rank)]["row"]["task"] for micro in range(self.gradient_accumulation_iters) for rank in range(self.num_replicas))
        task = self.dataset.samples[self.sample_index(absolute_microbatch)]["row"]["task"]
        return (self.num_replicas * self.gradient_accumulation_iters) / (len(counts) * counts[task])

    def microbatch_loss_scale(self, *, outer_iteration, stage, micro_step, fake_update_index=0):
        if self.official_pdmd:
            raise ValueError("Official PDMD must use the fresh microbatch cursor")
        if stage not in {"student", "fake"} or not 0 <= micro_step < self.gradient_accumulation_iters:
            raise ValueError("Invalid H3 DMD stage/micro_step")
        if stage == "fake" and not 0 <= fake_update_index < self.fake_update_ratio:
            raise ValueError("H3 DMD fake_update_index is out of range")
        if self.dmd_update_order == "student_first":
            role_offset = 0 if stage == "student" else 1 + fake_update_index
        else:
            role_offset = self.fake_update_ratio if stage == "student" else fake_update_index
        absolute = int(outer_iteration) * self.gradient_accumulation_iters * (1 + self.fake_update_ratio) + role_offset * self.gradient_accumulation_iters + micro_step
        return self.cursor_microbatch_loss_scale(absolute)

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0:
            raise ValueError("H3 curriculum loader epoch must be a non-negative integer")
        self.epoch = int(epoch)

    def __len__(self):
        return self.microbatches_per_round

    def __iter__(self):
        if self.start_microbatch is None:
            raise RuntimeError("Configure the H3 curriculum sampler before iteration")
        start = self.start_microbatch + self.epoch * len(self)
        return (self.sample_index(index) for index in range(start, start + len(self)))

    def checkpoint_metadata(self, *, gradient_accumulation_iters=None, fake_update_ratio=None):
        accum = self.gradient_accumulation_iters if gradient_accumulation_iters is None else int(gradient_accumulation_iters)
        ratio = self.fake_update_ratio if fake_update_ratio is None else int(fake_update_ratio)
        if accum is None or ratio is None:
            raise RuntimeError("H3 curriculum checkpoint metadata requires accumulation and fake ratio")
        if self.gradient_accumulation_iters is not None and self.gradient_accumulation_iters != accum:
            raise ValueError("H3 curriculum checkpoint accumulation differs from its configured sampler")
        # Checkpoint validation can run before configure() in ordinary DMD.
        if self.gradient_accumulation_iters is None:
            self.gradient_accumulation_iters = accum
        plan, offsets, repeats = self._plan(0)
        return {
            "route_mode": "new_then_old",
            "sampler": "h3_curriculum_full_coverage_v1",
            "kind": self.kind,
            "order": ["new", "old"],
            "repeat": "every_round",
            "source_rows": self.source_counts,
            "dataset_fingerprint": self.dataset.fingerprint,
            "data_parallel_world_size": self.num_replicas,
            "gradient_accumulation_iters": accum,
            "fake_update_ratio": ratio,
            "seed": self.seed,
            "update_order": self.dmd_update_order,
            "data_consumption_mode": "optimizer_update_critic_only" if self.official_pdmd else "all_role_updates",
            "student_data_source": "reuse_preceding_critic_trajectory" if self.official_pdmd else "fresh_microbatch",
            "resume_cursor_unit": "absolute_global_microbatch",
            "phase_microbatches": {key: stop - start for key, (start, stop) in offsets.items()},
            "repeated_rows_per_round": repeats,
            "microbatches_per_round": len(plan) // self.num_replicas,
            "old_policy": "task_round_robin_until_exhausted" if self.kind == "fl2av" else "all_image_counts_cost_local_batches",
            "balance_route_loss": self.balance_route_loss,
            "task_order": list(self.task_order),
            "cost_key": self.cost_key,
            "old_require_image_only": self.require_image_only,
            "old_image_counts": None if self.old_image_counts is None else list(self.old_image_counts),
            "padding_policy": "repeat_same_source_to_optimizer_boundary",
        }


def _source_paths(source, *, old, kind):
    paths = source["data_path"]
    values = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if old and kind == "fl2av" and any(Path(value).is_dir() for value in values):
        return _discover_minimax_h3_cache_metadata(
            roots=values,
            tasks=source.get("tasks", ["t2av", "i2av", "fl2av"]),
            buckets=source.get("buckets", ["landscape", "portrait"]),
            require_all_tasks=source.get("require_all_tasks", True),
            require_all_buckets=source.get("require_all_buckets", True),
            task_bucket_order=source.get("task_bucket_order"),
        )
    return values


@DATA_REGISTER("minimax_h3_curriculum_cache_dataset")
def build_minimax_h3_curriculum_cache_dataset(data_config, train_or_val="train"):
    if train_or_val != "train" or int(data_config.get("batch_size", 1)) != 1:
        raise ValueError("H3 curriculum supports train split with per-rank batch_size=1 only")
    if int(data_config.get("dataset_repeat", 1)) != 1:
        raise ValueError("H3 curriculum owns dataset repetition; dataset_repeat must be 1")
    kind = data_config.get("kind")
    sources = data_config.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"new", "old"}:
        raise ValueError("H3 curriculum sources must contain exactly new and old mappings")
    datasets = {}
    for name in ("new", "old"):
        source = sources[name]
        if not isinstance(source, dict) or "data_path" not in source:
            raise ValueError(f"H3 curriculum sources.{name} requires data_path")
        datasets[name] = LatentDataset(
            data_paths=_source_paths(source, old=name == "old", kind=kind),
            max_samples=source.get("max_samples"),
            defer_latent_loading=data_config.get("defer_latent_loading", True),
            prompt_column=source.get("prompt_column", "caption"),
            prompt_index=source.get("prompt_index", 0),
        )
    dataset = MiniMaxH3CurriculumCacheDataset(datasets["new"], datasets["old"])
    sampler = MiniMaxH3CurriculumSampler(
        dataset,
        kind=kind,
        seed=data_config.get("seed", 42),
        task_order=data_config.get("task_order", ["t2av", "i2av", "fl2av"]),
        balance_route_loss=data_config.get("balance_route_loss"),
        dmd_update_order=data_config.get("dmd_update_order", "student_first"),
        official_pdmd=data_config.get("official_pdmd", False),
        cost_key=sources["old"].get("cost_key", "packed_sequence_tokens_124"),
        require_image_only=sources["old"].get("require_image_only", True),
        old_image_counts=sources["old"].get("image_counts"),
    )
    return DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False, num_workers=data_config.get("num_workers", 8), pin_memory=data_config.get("pin_memory", True), drop_last=False)
