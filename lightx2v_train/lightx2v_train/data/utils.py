import csv
import json
import math
import random
import sys
from pathlib import Path

PROMPT_KEYS = ("prompt", "caption", "text")
CONDITION_KEYS = (
    "prompt_embed",
    "video_prompt_embeds",
    "audio_prompt_embeds",
    "prompt_embeds",
    "prompt_attention_mask",
    "video_context",
    "audio_context",
    "context_mask",
)


def resize_to_max_side(image, max_side):
    width, height = image.size
    if width >= height:
        new_width = max_side
        new_height = int(max_side * height / width)
    else:
        new_height = max_side
        new_width = int(max_side * width / height)
    return image.resize((new_width, new_height))


def resize_to_target_area(image, target_area):
    w, h = image.size
    ratio = w / h
    new_w = round(math.sqrt(target_area * ratio) / 16) * 16
    new_h = round(math.sqrt(target_area / ratio) / 16) * 16
    # Scale so that both dimensions are at least the target size, then crop.
    scale = max(new_w / w, new_h / h)
    scaled_w = round(w * scale)
    scaled_h = round(h * scale)
    image = image.resize((scaled_w, scaled_h), resample=3)  # BICUBIC=3
    # Center crop to exact (new_w, new_h)
    left = (scaled_w - new_w) // 2
    top = (scaled_h - new_h) // 2
    image = image.crop((left, top, left + new_w, top + new_h))
    return image


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
    return "" if value is None else str(value).strip()


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


def condition_payload(item):
    if isinstance(item, (list, tuple)):
        return _strip_leading_batch(item)
    if not isinstance(item, dict):
        raise TypeError(f"Condition payload must be a dict/list/tuple, got {type(item)!r}.")
    if "conditioning" in item:
        item = item["conditioning"]
    if "positive" in item:
        conditions = item["positive"]
    elif "conditions" in item:
        conditions = item["conditions"]
    elif "condition" in item:
        conditions = item["condition"]
    else:
        conditions = {key: item[key] for key in CONDITION_KEYS if key in item}
    if conditions is None or (isinstance(conditions, (dict, list, tuple)) and not conditions):
        raise KeyError("Condition payload must contain positive/conditions/condition or prompt embedding tensors.")
    return _strip_leading_batch(conditions)


def is_condition_payload(item):
    if isinstance(item, (list, tuple)):
        return True
    if not isinstance(item, dict):
        return False
    return bool({"conditioning", "positive", "conditions", "condition", *CONDITION_KEYS}.intersection(item.keys()))


def _strip_leading_batch(value):
    import torch

    if torch.is_tensor(value):
        if value.ndim >= 2 and value.shape[0] == 1:
            return value[0]
        return value
    if isinstance(value, dict):
        return {key: _strip_leading_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_leading_batch(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_leading_batch(item) for item in value)
    return value


def video_latent_payload(data):
    import torch

    if torch.is_tensor(data):
        return {"latents": data}
    if not isinstance(data, dict):
        return data
    normalized = dict(data)
    latents = normalized.get("latents")
    if not torch.is_tensor(latents) or latents.dim() != 2:
        return normalized
    num_frames = int(normalized["num_frames"])
    height = int(normalized["height"])
    width = int(normalized["width"])
    normalized["latents"] = latents.reshape(num_frames, height, width, latents.shape[-1]).permute(3, 0, 1, 2).contiguous()
    return normalized


def load_pt(path, weights_only=False):
    import torch

    return torch.load(path, map_location="cpu", weights_only=weights_only)


def crop_resize_exact(image, target_height, target_width, resampling="BILINEAR"):
    from PIL import Image

    width, height = image.size
    scale = max(target_width / width, target_height / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    if hasattr(Image, "Resampling"):
        resample = getattr(Image.Resampling, resampling)
    else:
        resample = getattr(Image, resampling)
    image = image.resize((resized_width, resized_height), resample)
    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return image.crop((left, top, left + target_width, top + target_height))


def frame_to_normalized_tensor(frame):
    import numpy as np
    import torch

    array = np.asarray(frame, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


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
    import imageio
    import torch
    from PIL import Image

    reader = imageio.get_reader(video_path)
    try:
        frames = []
        for frame_id in frame_sampler.frame_ids(reader):
            frame = Image.fromarray(reader.get_data(frame_id)).convert("RGB")
            frame = crop_resize_exact(frame, height, width)
            frames.append(frame_to_normalized_tensor(frame))
    finally:
        reader.close()
    return torch.stack(frames, dim=1)
