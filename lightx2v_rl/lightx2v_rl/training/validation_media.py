"""Task-separated image validation tables and original-reference exports."""

import json
import logging
import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from lightx2v_rl.paths import repository_path

logger = logging.getLogger(__name__)

REWARD_COLUMNS = {
    "unifiedreward_normalized": "unifiedreward",
    "unifiedreward_alignment": "ur_alignment",
    "unifiedreward_coherence": "ur_coherence",
    "unifiedreward_style": "ur_style",
    "unifiedreward_edit_success": "ur_edit_success",
    "unifiedreward_preservation": "ur_preservation",
    "unifiedreward_format_error": "ur_format_error",
    "hpsv3_general_raw": "hpsv3_raw",
}


def _image_from_tensor(value):
    """Use the rollout's uint8 CHW pixels without another model preprocessing."""
    pixels = value.detach().cpu().numpy()
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[0] not in (1, 3, 4):
        raise ValueError("Image validation expects uint8 CHW rollout responses")
    pixels = np.moveaxis(pixels, 0, -1)
    return Image.fromarray(pixels[..., 0] if pixels.shape[-1] == 1 else pixels).copy()


def _original_image(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).copy()
    # PNG preserves source transparency; uncommon JPEG modes need conversion.
    return image if image.mode in ("1", "L", "LA", "RGB", "RGBA", "P", "I", "I;16") else image.convert("RGB")


def _preview_group(task, sources):
    if task not in {"t2i", "i2i"} or (task == "t2i" and sources) or (task == "i2i" and not sources):
        raise ValueError("Preview samples require a valid task and matching reference images")
    return "t2i" if task == "t2i" else ("i2i_single" if len(sources) == 1 else "i2i_multi")


def _load_preview(path):
    selected = {}
    with Path(repository_path(path)).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in selected:
                raise ValueError(f"Preview sample IDs must be nonempty, unique strings: {sample_id!r}")
            sources = row.get("source_images", [])
            if not isinstance(sources, list) or any(not isinstance(source, str) or not source for source in sources):
                raise ValueError(f"Preview source_images must be a list of nonempty paths: {sample_id}")
            selected[sample_id] = (len(selected), _preview_group(row.get("task_type"), sources), len(sources))
    if not selected:
        raise ValueError("Validation preview file contains no samples")
    return selected


class TaskValidationMedia:
    """Collect a bounded, stable sample per task without truncating evaluation.

    A fresh collector and fresh W&B tables are created for every evaluation.
    A configured preview file selects all its sample IDs, in manifest order,
    without applying the per-task budget. Otherwise selection uses the dataset
    ordinal in ``extra_info`` rather than TQ completion order. The original
    references are read from the dataset, never decoded from model latents.
    """

    def __init__(self, config, global_step):
        self.config = config
        self.global_step = int(global_step)
        self.limit = config.trainer.get("validation_samples_per_task", 8)
        if isinstance(self.limit, bool) or not isinstance(self.limit, Integral) or self.limit < 0:
            raise ValueError("trainer.validation_samples_per_task must be a nonnegative integer")
        preview_file = config.trainer.get("validation_preview_file")
        self.preview = _load_preview(preview_file) if preview_file else None
        groups = ("t2i", "i2i_single", "i2i_multi") if self.preview is not None else ("t2i", "i2i")
        self.samples = {group: {} for group in groups}

    def validate_dataset(self, dataset):
        """Fail startup if a preview ID is absent or its reference role differs."""
        if self.preview is None:
            return {}
        rows = {str(row["id"]): row for row in dataset.rows}
        counts = {group: 0 for group in self.samples}
        for sample_id, (_, group, source_count) in self.preview.items():
            if sample_id not in rows:
                raise ValueError(f"Validation preview ID is absent from the full validation dataset: {sample_id}")
            row = rows[sample_id]
            sources = row.get("source_images", [])
            if _preview_group(row.get("task_type"), sources) != group or len(sources) != source_count:
                raise ValueError(f"Validation preview reference roles differ from the validation dataset: {sample_id}")
            counts[group] += 1
        return counts

    def collect(self, data, inputs, reward_extra_info):
        if self.preview is None and not self.limit:
            return
        extras = data.non_tensor_batch.get("extra_info")
        if extras is None:
            raise ValueError("Task validation media requires dataset extra_info metadata")
        if len(extras) != len(inputs) or len(data.batch["responses"]) != len(inputs):
            raise ValueError("Validation media, prompts and metadata must be aligned")
        for position, extra in enumerate(extras):
            if not isinstance(extra, Mapping):
                raise TypeError("Validation extra_info rows must be mappings")
            task = extra.get("task_type")
            if task not in {"t2i", "i2i"}:
                raise ValueError(f"Unsupported image validation task: {task!r}")
            ordinal = extra.get("index")
            sample_id = str(extra.get("sample_id", ordinal))
            if self.preview is not None and sample_id not in self.preview:
                continue
            sources = list(extra.get("source_images", []))
            if (task == "t2i" and sources) or (task == "i2i" and not sources):
                raise ValueError("Image validation requires no T2I references and at least one I2I reference")
            if self.preview is not None:
                ordinal, group, source_count = self.preview[sample_id]
                if _preview_group(task, sources) != group or len(sources) != source_count:
                    raise ValueError(f"Validation result differs from preview reference roles: {sample_id}")
            else:
                group = task
                if isinstance(ordinal, bool) or not isinstance(ordinal, Integral):
                    raise ValueError("Stable validation media selection requires extra_info.index")
            key = (int(ordinal), sample_id)
            selected = self.samples[group]
            if key in selected or (self.preview is None and len(selected) >= self.limit and key >= max(selected)):
                continue
            rewards = {}
            for name, label in REWARD_COLUMNS.items():
                column = reward_extra_info.get(name)
                if column is None:
                    continue
                value = column[position]
                if isinstance(value, Real) and math.isfinite(value):
                    rewards[label] = float(value)
            selected[key] = {
                "sample_id": sample_id,
                "prompt": extra.get("prompt", inputs[position]),
                "source_paths": sources,
                "generated_image": _image_from_tensor(data.batch["responses"][position]),
                "rewards": rewards,
            }
            if self.preview is None and len(selected) > self.limit:
                del selected[max(selected)]

    def log(self):
        if self.preview is not None:
            collected = {sample["sample_id"] for selected in self.samples.values() for sample in selected.values()}
            missing = self.preview.keys() - collected
            if missing:
                raise ValueError(f"Missing {len(missing)} validation preview results: {', '.join(sorted(missing)[:8])}")
        if not any(self.samples.values()):
            return
        directory = self.config.trainer.get("validation_data_dir")
        backends = self.config.trainer.get("logger", [])
        wandb_enabled = "wandb" in ([backends] if isinstance(backends, str) else backends)
        wandb = None
        if wandb_enabled:
            import wandb

            if wandb.run is None:
                logger.warning("Validation media cannot reach W&B: no active run")
                wandb = None
        payload = {}
        for task, selected in self.samples.items():
            rows = []
            for key in sorted(selected):
                sample = selected[key]
                try:
                    sources = [_original_image(path) for path in sample["source_paths"]]
                except OSError as error:
                    if self.preview is not None:
                        raise OSError(f"Cannot visualize preview references for {sample['sample_id']}") from error
                    logger.warning("Skipping validation source %s: %s", sample["sample_id"], error)
                    continue
                rows.append({**sample, "source_images": sources})
            if not rows:
                continue
            if directory:
                self._dump(Path(directory) / f"global_step_{self.global_step}" / task, rows)
            if wandb is not None:
                reward_keys = [label for label in REWARD_COLUMNS.values() if any(label in row["rewards"] for row in rows)]
                columns = ["iteration", "sample_id", "prompt"]
                source_count = max(len(row["source_images"]) for row in rows)
                if task != "t2i":
                    columns.append("source_image")
                    columns.extend(f"reference_image_{index + 1}" for index in range(1, source_count))
                columns += ["generated_image", *reward_keys]
                table = wandb.Table(columns=columns)
                for row in rows:
                    values = [self.global_step, row["sample_id"], row["prompt"]]
                    if task != "t2i":
                        for index in range(source_count):
                            source = row["source_images"][index] if index < len(row["source_images"]) else None
                            caption = "Image 1: original canvas" if index == 0 else f"Image {index + 1}: reference"
                            values.append(wandb.Image(source, caption=caption) if source is not None else None)
                    values.append(wandb.Image(row["generated_image"], caption=row["prompt"]))
                    values.extend(row["rewards"].get(name) for name in reward_keys)
                    table.add_data(*values)
                payload[f"val/{task}"] = table
            logger.info("Validation media step=%d task=%s samples=%d", self.global_step, task, len(rows))
        if payload:
            # The trainer logs scalars at this same step immediately afterwards.
            wandb.log(payload, step=self.global_step, commit=False)

    def _dump(self, directory, rows):
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        for index, row in enumerate(rows):
            generated = f"{index:02d}_generated.png"
            row["generated_image"].save(directory / generated)
            sources = []
            for source_index, image in enumerate(row["source_images"]):
                source = f"{index:02d}_source.png" if source_index == 0 else f"{index:02d}_reference_{source_index + 1}.png"
                image.save(directory / source)
                sources.append(source)
            records.append(
                {
                    "iteration": self.global_step,
                    "sample_id": row["sample_id"],
                    "prompt": row["prompt"],
                    "source_image": sources[0] if sources else None,
                    "source_images": sources,
                    "generated_image": generated,
                    "rewards": row["rewards"],
                }
            )
        with (directory / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
