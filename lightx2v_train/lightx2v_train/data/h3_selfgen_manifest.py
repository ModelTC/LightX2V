"""Per-record H3 geometry and the two pools in the organized FL2AV manifest."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from lightx2v_train.runtime.distributed import get_data_parallel_rank, get_data_parallel_world_size
from lightx2v_train.utils.registry import DATA_REGISTER
from lightx2v_train.trainers.turbo.cursor import TurboCursorSampler


def read_pools(path):
    raw = Path(path).read_bytes()
    prompt_only, paired = [], []
    for row in json.loads(raw):
        if row["task"] not in ("t2av", "i2av") or row["num_frames"] != 124:
            continue
        if not row["teacher_video"]:
            prompt_only.append(row)
        elif row["teacher_video_latent"] and row["teacher_audio_latent"]:
            paired.append(row)
    return prompt_only, paired, hashlib.sha256(raw).hexdigest()


def load_positive(row):
    return torch.load(row["condition"], map_location="cpu", weights_only=True)["conditioning"]["positive"]


class PDMDManifest(Dataset):
    def __init__(self, path, *, paired_only=False):
        prompt_only, paired, self.digest = read_pools(path)
        if paired_only:
            prompt_only = []
        self.prompt_count = len(prompt_only)
        self.samples = prompt_only + paired

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        return {
            "conditioning": {"positive": load_positive(row), "prompt": row["training_prompt"]},
            "inputs": {},
            "meta": {"sample_id": row["sample_id"], "target_height": row["height"], "target_width": row["width"],
                     "num_frames": row["num_frames"], "video_latent_path": row["teacher_video_latent"],
                     "audio_latent_path": row["teacher_audio_latent"]},
        }


class PhaseSampler(TurboCursorSampler):
    """Shuffle each pool globally. DataLoader prefetch never changes the phase boundary."""

    def __init__(self, dataset, start_gan, **kwargs):
        super().__init__(dataset, **kwargs)
        self.start_gan = start_gan

    def configure_from_microbatch_offset(self, *, start_microbatch, gradient_accumulation_iters, fake_update_ratio):
        super().configure_from_microbatch_offset(start_microbatch=start_microbatch)
        critic_updates = self.start_gan - self.start_gan // (fake_update_ratio + 1)
        self.switch = critic_updates * gradient_accumulation_iters

    def __iter__(self):
        start = self.start_microbatch + self.epoch * len(self)
        last = None
        for cursor in range(start, start + len(self)):
            paired = cursor >= self.switch
            offset = self.dataset.prompt_count if paired else 0
            size = len(self.dataset) - offset if paired else self.dataset.prompt_count
            relative = cursor - self.switch if paired else cursor
            epoch, pos = divmod(relative * self.num_replicas + self.rank, size)
            if (paired, epoch) != last:
                gen = torch.Generator().manual_seed(self.seed + 2 * epoch + int(paired))
                order = torch.randperm(size, generator=gen).tolist()
                last = paired, epoch
            yield offset + order[pos]


@DATA_REGISTER("h3_selfgen_manifest")
def build_manifest_loader(data_config, train_or_val="train", unconditional_prompt=" ", sample_processor=None):
    start = data_config["start_gan"]
    dataset = PDMDManifest(data_config["data_path"], paired_only=start == 0)
    options = dict(seed=data_config["seed"], num_replicas=get_data_parallel_world_size(), rank=get_data_parallel_rank())
    sampler = PhaseSampler(dataset, start, **options) if start else TurboCursorSampler(dataset, **options)
    # Each item is already a native single-sample H3 batch.
    return DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=data_config.get("num_workers", 2))
