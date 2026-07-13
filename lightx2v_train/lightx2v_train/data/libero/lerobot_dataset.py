import bisect
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .video_decoder import decode_video_frames


@dataclass(frozen=True)
class Episode:
    root: Path
    index: int
    length: int
    data_path: str
    video_path: str
    chunks_size: int
    tasks: dict[int, str]

    def parquet_path(self) -> Path:
        return self.root / self.data_path.format(
            episode_chunk=self.index // self.chunks_size,
            episode_index=self.index,
        )

    def camera_path(self, key: str) -> Path:
        return self.root / self.video_path.format(
            episode_chunk=self.index // self.chunks_size,
            episode_index=self.index,
            video_key=key,
        )


def _read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _column_to_tensor(column: pa.ChunkedArray) -> torch.Tensor:
    column = column.combine_chunks()
    if pa.types.is_list(column.type) or pa.types.is_fixed_size_list(column.type):
        array = np.asarray(column.to_pylist())
    else:
        array = column.to_numpy(zero_copy_only=False)
    return torch.from_numpy(np.array(array, copy=True))


class LiberoLeRobotDataset(Dataset):
    """Minimal local reader for FastWAM's processed LIBERO LeRobot v2.1 data."""

    def __init__(
        self,
        dataset_dirs,
        image_keys,
        state_key,
        action_key,
        num_frames,
        global_sample_stride=1,
        val_set_proportion=0.0,
        is_training_set=True,
        seed=42,
        video_backend=None,
    ):
        if num_frames < 2:
            raise ValueError(f"num_frames must be at least 2, got {num_frames}")
        self.image_keys = list(image_keys)
        self.state_key = state_key
        self.action_key = action_key
        self.num_frames = int(num_frames)
        self.global_sample_stride = int(global_sample_stride)
        self.video_backend = video_backend

        episodes = []
        fps_values = set()
        for dataset_dir in dataset_dirs:
            root = Path(dataset_dir).expanduser().resolve()
            info = _read_json(root / "meta" / "info.json")
            tasks = {int(item["task_index"]): item["task"] for item in _read_jsonl(root / "meta" / "tasks.jsonl")}
            episode_meta = _read_jsonl(root / "meta" / "episodes.jsonl")
            fps_values.add(int(info["fps"]))
            self._validate_features(root, info["features"])

            indices = list(range(len(episode_meta)))
            if val_set_proportion >= 1e-6:
                rng = np.random.default_rng(seed)
                rng.shuffle(indices)
                split = int(len(indices) * (1.0 - val_set_proportion))
                indices = indices[:split] if is_training_set else indices[split:]

            by_index = {int(item["episode_index"]): item for item in episode_meta}
            for episode_index in indices:
                item = by_index[episode_index]
                episodes.append(
                    Episode(
                        root=root,
                        index=episode_index,
                        length=int(item["length"]),
                        data_path=info["data_path"],
                        video_path=info["video_path"],
                        chunks_size=int(info["chunks_size"]),
                        tasks=tasks,
                    )
                )

        if not episodes:
            raise RuntimeError("No LIBERO episodes were selected.")
        if len(fps_values) != 1:
            raise ValueError(f"All LIBERO datasets must have the same fps, got {sorted(fps_values)}")
        self.fps = fps_values.pop()
        self.episodes = episodes
        self._episode_ends = np.cumsum([episode.length for episode in episodes]).tolist()

    def _validate_features(self, root, features):
        required = [*self.image_keys, self.state_key, self.action_key, "timestamp", "task_index"]
        missing = [key for key in required if key not in features]
        if missing:
            raise ValueError(f"LIBERO dataset {root} is missing features: {missing}")
        non_video = [key for key in self.image_keys if features[key].get("dtype") != "video"]
        if non_video:
            raise ValueError(f"LIBERO camera features must be videos, got: {non_video}")

    def __len__(self):
        return self._episode_ends[-1]

    @lru_cache(maxsize=32)
    def _load_episode(self, episode_position):
        episode = self.episodes[episode_position]
        table = pq.read_table(episode.parquet_path())
        return {key: _column_to_tensor(table[key]) for key in table.column_names}

    def _locate(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self)}")
        episode_position = bisect.bisect_right(self._episode_ends, index)
        episode_start = 0 if episode_position == 0 else self._episode_ends[episode_position - 1]
        return episode_position, index - episode_start

    def _window(self, values, frame_index, length):
        offsets = torch.arange(length, dtype=torch.long) * self.global_sample_stride
        indices = frame_index + offsets
        is_pad = indices >= values.shape[0]
        indices.clamp_(max=values.shape[0] - 1)
        return values[indices], is_pad, indices

    def __getitem__(self, index):
        episode_position, frame_index = self._locate(index)
        episode = self.episodes[episode_position]
        data = self._load_episode(episode_position)

        state, state_is_pad, observation_indices = self._window(data[self.state_key], frame_index, self.num_frames)
        action, action_is_pad, _ = self._window(data[self.action_key], frame_index, self.num_frames - 1)
        timestamps = data["timestamp"][observation_indices].float().tolist()
        tolerance = max(1e-4, 1.0 / self.fps - 1e-4)
        images = {
            key: decode_video_frames(
                episode.camera_path(key),
                timestamps,
                tolerance_s=tolerance,
                backend=self.video_backend,
            )
            for key in self.image_keys
        }

        task_index = int(data["task_index"][frame_index].item())
        return {
            "idx": int(index),
            "task": episode.tasks[task_index],
            "action": {"default": action.float()},
            "state": {"default": state.float()},
            "images": {key.removeprefix("observation.images."): value for key, value in images.items()},
            "action_is_pad": action_is_pad,
            "state_is_pad": state_is_pad,
            "image_is_pad": state_is_pad.clone(),
        }
