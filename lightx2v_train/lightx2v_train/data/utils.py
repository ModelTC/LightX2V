import csv
import json
import math
import random
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image

PROMPT_KEYS = ("prompt", "caption", "text")
_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def to_list(value):
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def record_value(record, *keys, default=None):
    for key in keys:
        if isinstance(record, dict) and key in record and record[key] not in (None, ""):
            return record[key]
    return default


def prompt_text(record, prompt_column="prompt", prompt_index=0):
    if isinstance(record, dict):
        value = record_value(record, prompt_column, *PROMPT_KEYS, default="")
    elif isinstance(record, (list, tuple)):
        value = record[prompt_index] if prompt_index < len(record) else ""
    else:
        value = record
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def read_records(path, prompt_column="prompt", prompt_index=0):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".list"}:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                prompt = line.strip()
                if prompt:
                    yield {"prompt": prompt}
        return

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield _normalize_record(json.loads(line), prompt_index)
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if isinstance(records, dict):
            records = records.get("samples", records.get("items", records.get("prompts", records.get("data", [records]))))
        for record in records:
            yield _normalize_record(record, prompt_index)
        return

    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
        if first_row is None:
            return
        header = [column.strip() for column in first_row]
        known_columns = {
            prompt_column,
            *PROMPT_KEYS,
            "video",
            "video_path",
            "audio",
            "audio_path",
            "image",
            "image_path",
            "video_latent_path",
            "audio_latent_path",
            "condition_path",
            "negative_condition_path",
        }
        if set(header).intersection(known_columns):
            for row in reader:
                yield {key: row[idx] if idx < len(row) else "" for idx, key in enumerate(header)}
        else:
            yield _csv_prompt_record(first_row, prompt_index)
            for row in reader:
                yield _csv_prompt_record(row, prompt_index)


def _normalize_record(record, prompt_index=0):
    if isinstance(record, str):
        return {"prompt": record.strip()}
    if isinstance(record, (list, tuple)):
        return _csv_prompt_record(record, prompt_index)
    if not isinstance(record, dict):
        return {"prompt": str(record).strip()}
    return dict(record)


def _csv_prompt_record(row, prompt_index=0):
    prompt = str(row[prompt_index]).strip() if prompt_index < len(row) else ""
    return {"prompt": prompt}


def resolve_data_path(value, base_dir, roots=(), subdirs=()):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path

    candidates = [Path(base_dir) / raw_path]
    candidates.extend(Path(base_dir) / subdir / raw_path.name for subdir in subdirs)
    for root in roots:
        if root is not None:
            root = Path(root)
            candidates.extend([root / raw_path, root / raw_path.name])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


class VideoFrameSampler:
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_rate=24,
        fix_frame_rate=False,
        random_start=False,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_rate = frame_rate
        self.fix_frame_rate = fix_frame_rate
        self.random_start = random_start

    def available_frames(self, reader):
        total_raw_frames = int(reader.count_frames())
        if not self.fix_frame_rate:
            return total_raw_frames
        meta_data = reader.get_meta_data()
        duration = meta_data.get("duration") or total_raw_frames / meta_data["fps"]
        return int(math.floor(duration * self.frame_rate))

    def sample_count(self, reader):
        num_frames = self.num_frames
        total_frames = self.available_frames(reader)
        if total_frames < num_frames:
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return max(1, num_frames)

    def raw_frame_id(self, sequence_id, raw_frame_rate, total_raw_frames):
        if not self.fix_frame_rate:
            return sequence_id
        target_time = sequence_id / self.frame_rate
        frame_id = int(round(target_time * raw_frame_rate))
        return min(frame_id, total_raw_frames - 1)

    def frame_ids(self, reader):
        raw_frame_rate = reader.get_meta_data().get("fps", self.frame_rate)
        total_raw_frames = int(reader.count_frames())
        num_frames = self.sample_count(reader)
        max_start = max(0, self.available_frames(reader) - num_frames)
        start = random.randint(0, max_start) if self.random_start and max_start > 0 else 0
        return [self.raw_frame_id(start + frame_id, raw_frame_rate, total_raw_frames) for frame_id in range(num_frames)]


def load_video_tensor(video_path, height, width, frame_sampler):
    reader = imageio.get_reader(video_path)
    try:
        frames = []
        for frame_id in frame_sampler.frame_ids(reader):
            frame = Image.fromarray(reader.get_data(frame_id)).convert("RGB")
            frame_width, frame_height = frame.size
            scale = max(width / frame_width, height / frame_height)
            resized_width = round(frame_width * scale)
            resized_height = round(frame_height * scale)
            frame = frame.resize((resized_width, resized_height), _BILINEAR)
            left = max(0, (resized_width - width) // 2)
            top = max(0, (resized_height - height) // 2)
            frame = frame.crop((left, top, left + width, top + height))
            array = np.asarray(frame, dtype=np.float32) / 127.5 - 1.0
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
    finally:
        reader.close()
    return torch.stack(frames, dim=1)
