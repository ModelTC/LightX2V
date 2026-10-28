"""Independent full-target T2AV cache for H3 Turbo consistency objectives.

CM has its own absolute global-microbatch cursor. The sampler concatenates
deterministic shuffled data epochs without truncation or padding: a global
batch may cross a data-epoch boundary. Loader epochs are fixed-size segments
of that stream, so an arbitrary checkpoint resumes its exact suffix.
"""

import math
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Sampler

from lightx2v_train.data.minimax_h3_cache_dataset import LatentDataset, _condition_payload
from lightx2v_train.data.minimax_h3_turbo_paths import load_cm_records
from lightx2v_train.runtime.distributed import get_data_parallel_rank, get_data_parallel_world_size
from lightx2v_train.utils.registry import DATA_REGISTER


class MiniMaxH3TurboCMDataset(LatentDataset):
    """Reuse the existing full AV loader and map three cache paths in memory."""

    def __init__(self, metadata_path, *, path_mapping=None, tasks=None, max_samples=None, defer_latent_loading=True):
        self.metadata_path, self._cm_rows, self.metadata_fingerprint = load_cm_records(metadata_path, path_mapping, tasks, check_files=False)
        super().__init__(str(self.metadata_path), max_samples=max_samples, defer_latent_loading=defer_latent_loading)

    def _index_metadata(self, metadata_path):
        for row in self._cm_rows:
            self.samples.append({"type": "metadata", "row": row, "base_dir": str(metadata_path.parent)})

    def _validate_condition_item(self, item, row, condition_path):
        if not isinstance(item, dict) or item.get("cache_fingerprint") != row["cache_fingerprint"]:
            raise ValueError(f"CM condition fingerprint differs from its target manifest: {condition_path}")
        if str(item.get("source_id")) != str(row["source_id"]) or item.get("task") != "t2av":
            raise ValueError(f"CM condition identity/task differs from its target manifest: {condition_path}")
        positive = _condition_payload(item)
        if not isinstance(positive, dict) or positive.get("task", "t2av") != "t2av":
            raise ValueError(f"CM requires actual T2AV condition payloads: {condition_path}")
        tags = positive.get("text_token_tags")
        if not torch.is_tensor(tags) or not bool((tags == 1).all()):
            raise ValueError(f"CM T2AV condition must have text-only token tags: {condition_path}")

    def _load_metadata_sample(self, row, base_dir):
        sample = super()._load_metadata_sample(row, base_dir)
        sample["meta"].update(source_id=row["source_id"], cache_fingerprint=row["cache_fingerprint"], dataset_fingerprint=self.metadata_fingerprint)
        return sample


class MiniMaxH3TurboCMSampler(Sampler):
    """Every rank takes one item from an infinite shuffled global stream."""

    def __init__(self, dataset, *, num_replicas=None, rank=None, seed=42, shuffle=True):
        self.dataset = dataset
        self.num_replicas = get_data_parallel_world_size() if num_replicas is None else int(num_replicas)
        self.rank = get_data_parallel_rank() if rank is None else int(rank)
        self.seed, self.shuffle = int(seed), bool(shuffle)
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("CM sampler requires a positive world size and a valid rank.")
        self.num_samples = len(dataset)
        if self.num_samples < 1:
            raise ValueError("CM sampler requires a non-empty full-target dataset.")
        self.microbatches_per_epoch = math.ceil(self.num_samples / self.num_replicas)
        self.epoch = 0
        self.start_microbatch = None
        self.gradient_accumulation_iters = self.fake_update_ratio = None
        self._permutations = OrderedDict()

    def configure_from_microbatch_offset(self, start_microbatch, gradient_accumulation_iters=1, fake_update_ratio=1, **kwargs):
        del kwargs  # The independent CM stream does not consume fake-role data.
        if int(start_microbatch) != start_microbatch or start_microbatch < 0:
            raise ValueError("CM start_microbatch must be a non-negative integer.")
        if int(gradient_accumulation_iters) < 1 or int(fake_update_ratio) < 1:
            raise ValueError("CM accumulation and fake ratio must be positive.")
        self.start_microbatch = int(start_microbatch)
        self.gradient_accumulation_iters = int(gradient_accumulation_iters)
        self.fake_update_ratio = int(fake_update_ratio)
        self.epoch = 0

    def set_epoch(self, epoch):
        if int(epoch) != epoch or epoch < 0:
            raise ValueError("CM loader epoch must be a non-negative integer.")
        self.epoch = int(epoch)

    def _permutation(self, epoch):
        if epoch not in self._permutations:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self._permutations[epoch] = torch.randperm(self.num_samples, generator=generator).tolist() if self.shuffle else list(range(self.num_samples))
            while len(self._permutations) > 2:
                self._permutations.popitem(last=False)
        self._permutations.move_to_end(epoch)
        return self._permutations[epoch]

    def sample_index(self, absolute_microbatch):
        if int(absolute_microbatch) != absolute_microbatch or absolute_microbatch < 0:
            raise ValueError("CM absolute microbatch must be a non-negative integer.")
        data_epoch, offset = divmod(int(absolute_microbatch) * self.num_replicas + self.rank, self.num_samples)
        return self._permutation(data_epoch)[offset]

    def __iter__(self):
        if self.start_microbatch is None:
            raise RuntimeError("Configure the CM sampler with an absolute microbatch cursor before iteration.")
        start = self.start_microbatch + self.epoch * self.microbatches_per_epoch
        return (self.sample_index(index) for index in range(start, start + self.microbatches_per_epoch))

    def __len__(self):
        return self.microbatches_per_epoch

    @staticmethod
    def cursor_microbatch_loss_scale(absolute_microbatch):
        del absolute_microbatch
        return 1.0

    def checkpoint_metadata(self, *, gradient_accumulation_iters=None, fake_update_ratio=None):
        return {
            "sampler": "h3_turbo_cm_continuous_shuffle_v1",
            "metadata_fingerprint": self.dataset.metadata_fingerprint,
            "num_samples": self.num_samples,
            "data_parallel_world_size": self.num_replicas,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "orientation_sampling": "natural",
            "remainder_policy": "continue_next_shuffled_data_epoch",
            "resume_cursor_unit": "absolute_global_microbatch",
            "data_consumption_mode": "independent_cm_full_target_av",
            "gradient_accumulation_iters": self.gradient_accumulation_iters if gradient_accumulation_iters is None else int(gradient_accumulation_iters),
            "fake_update_ratio": self.fake_update_ratio if fake_update_ratio is None else int(fake_update_ratio),
        }


@DATA_REGISTER("minimax_h3_turbo_cm_dataset")
def build_minimax_h3_turbo_cm_dataset(data_config_split, train_or_val="cm"):
    if train_or_val not in {"cm", "train"}:
        raise ValueError("The independent Turbo CM loader supports only the cm/train split.")
    if int(data_config_split.get("batch_size", 1)) != 1:
        raise ValueError("H3 Turbo CM requires per-rank batch_size=1.")
    if int(data_config_split.get("dataset_repeat", 1)) != 1:
        raise ValueError("CM sampler owns epoch cycling; dataset_repeat must be 1.")
    if data_config_split.get("drop_last", False):
        raise ValueError("H3 Turbo CM uses continuous epoch cycling; drop_last must be false.")
    paths = data_config_split["data_path"]
    if isinstance(paths, (tuple, list)):
        if len(paths) != 1:
            raise ValueError("CM requires one full-target metadata.jsonl manifest.")
        paths = paths[0]
    max_samples = data_config_split.get("max_samples")
    if max_samples is not None and int(max_samples) < 1:
        raise ValueError("CM max_samples must be positive.")
    dataset = MiniMaxH3TurboCMDataset(
        paths,
        path_mapping=data_config_split.get("path_mapping"),
        tasks=data_config_split.get("tasks", ["t2av"]),
        max_samples=max_samples,
        defer_latent_loading=bool(data_config_split.get("defer_latent_loading", True)),
    )
    sampler = MiniMaxH3TurboCMSampler(dataset, seed=data_config_split.get("seed", 42), shuffle=data_config_split.get("shuffle", True))
    return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=int(data_config_split.get("num_workers", 0)), pin_memory=bool(data_config_split.get("pin_memory", False)), drop_last=False)
