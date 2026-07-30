import hashlib
import os
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset

from .lerobot_dataset import LiberoLeRobotDataset
from .normalizer import load_dataset_stats

PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


class RobotVideoDataset(Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        processor,
        text_embedding_cache_dir,
        pretrained_norm_stats,
        num_frames=33,
        context_len=128,
        val_set_proportion=0.0,
        is_training_set=True,
        global_sample_stride=1,
        action_video_freq_ratio=4,
        skip_padding_as_possible=False,
        max_padding_retry=3,
        video_backend=None,
    ):
        image_keys = [f"observation.images.{item['key']}" for item in shape_meta["images"]]
        self.lerobot_dataset = LiberoLeRobotDataset(
            dataset_dirs=dataset_dirs,
            image_keys=image_keys,
            state_key="observation.state",
            action_key="action",
            num_frames=num_frames,
            global_sample_stride=global_sample_stride,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            video_backend=video_backend,
        )
        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        if (self.num_frames - 1) % self.action_video_freq_ratio:
            raise ValueError("num_frames - 1 must be divisible by action_video_freq_ratio")
        if ((self.num_frames - 1) // self.action_video_freq_ratio) % 4:
            raise ValueError("The number of future video frames must be divisible by the VAE temporal factor 4")
        self.video_sample_indices = list(range(0, self.num_frames, self.action_video_freq_ratio))
        self.context_len = int(context_len)
        self.text_embedding_cache_dir = os.path.abspath(os.path.expanduser(text_embedding_cache_dir))
        self.skip_padding_as_possible = bool(skip_padding_as_possible)
        self.max_padding_retry = int(max_padding_retry)

        stats_path = os.path.abspath(os.path.expanduser(pretrained_norm_stats))
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"LIBERO normalization stats do not exist: {stats_path}")
        processor.set_normalizer_from_stats(load_dataset_stats(stats_path))
        self.processor = processor

    def __len__(self):
        return len(self.lerobot_dataset)

    def _sample_without_padding(self, index):
        for attempt in range(self.max_padding_retry + 1):
            sample = self.processor.preprocess(self.lerobot_dataset[index])
            if not self.skip_padding_as_possible:
                return sample
            has_padding = any(bool(sample[key].any()) for key in ("action_is_pad", "image_is_pad", "proprio_is_pad"))
            if not has_padding or attempt == self.max_padding_retry:
                return sample
            index = np.random.randint(len(self))
        raise AssertionError("unreachable")

    def _cached_context(self, prompt):
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = os.path.join(
            self.text_embedding_cache_dir,
            f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt",
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing cached text embedding: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        context = payload["context"]
        mask = payload["mask"].bool()
        if context.ndim != 2 or context.shape[0] != self.context_len:
            raise ValueError(f"Invalid cached context shape {tuple(context.shape)} in {path}")
        if mask.shape != (self.context_len,):
            raise ValueError(f"Invalid cached context mask shape {tuple(mask.shape)} in {path}")
        context[~mask] = 0.0
        return context, torch.ones_like(mask)

    def _get(self, index):
        sample = self._sample_without_padding(index)
        video = sample["pixel_values"][:, self.video_sample_indices]
        if video.shape[0] != 2:
            raise ValueError(f"FastWAM LIBERO expects two cameras, got {video.shape[0]}")
        video = torch.cat((video[0], video[1]), dim=-1)
        video = video.mul(2.0).sub(1.0).permute(1, 0, 2, 3).contiguous()

        action = sample["action"]
        proprio = sample["proprio"][:-1]
        prompt = PROMPT_TEMPLATE.format(task=sample["instruction"])
        context, context_mask = self._cached_context(prompt)
        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": sample["image_is_pad"][self.video_sample_indices],
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }

    def __getitem__(self, index):
        try:
            return self._get(index)
        except Exception:
            traceback.print_exc()
            fallback = np.random.randint(len(self))
            return self._get(fallback)
