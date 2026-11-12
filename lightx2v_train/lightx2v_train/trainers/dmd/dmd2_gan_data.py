"""Independent, restartable positive-data stream for the H3 DMD2 GAN objective.

Manifest preflight uses only the standard library. Tensors are loaded through
the existing H3 LatentDataset/capability only when a critic consumes a sample.
The caller checkpoints its absolute consumed-critic-microbatch count; this
stream has no iterator or worker state to restore.
"""

import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path

CACHE_PATH_KEYS = ("condition_path", "video_latent_path", "audio_latent_path")
DEFAULT_GEOMETRY = (124, 768, 1344)


def _has_value(value):
    return value is not None and (not isinstance(value, (str, list, tuple, dict)) or len(value) > 0)


def _dimension(value):
    result = int(value)
    if isinstance(value, bool) or float(value) != result:
        raise ValueError("Target dimensions must be integers.")
    return result


def _check_t2av_condition(condition, location):
    if not isinstance(condition, dict) or condition.get("task") != "t2av":
        raise ValueError(f"GAN real data requires an actual T2AV condition: {location}")
    for key in ("references", "keyframe_anchors", "keyframe_latents", "negative", "negative_conditions"):
        if _has_value(condition.get(key)):
            raise ValueError(f"GAN real T2AV condition contains unexpected {key}: {location}")


def _check_ref_condition(condition, location):
    if not isinstance(condition, dict) or str(condition.get("task", "")).lower() not in {"ref2av", "ref2va"}:
        raise ValueError(f"GAN real data requires an actual Ref2AV condition: {location}")
    for key in ("keyframe_anchors", "keyframe_latents", "negative", "negative_conditions", "condition_audio_latents"):
        if _has_value(condition.get(key)):
            raise ValueError(f"GAN real image-only Ref condition contains unexpected {key}: {location}")
    references = condition.get("references")
    if not isinstance(references, (list, tuple)) or not 1 <= len(references) <= 9:
        raise ValueError(f"GAN real Ref condition requires ordered image references: {location}")
    for reference in references:
        get = reference.get if isinstance(reference, dict) else lambda key, default=None: getattr(reference, key, default)
        if str(get("kind", "")).lower() != "image" or _has_value(get("audio_latents")) or get("num_audio_latents", 0) != 0:
            raise ValueError(f"GAN real Ref condition must contain image-only references: {location}")
        if isinstance(reference, dict) and reference.get("normalized") is not True:
            raise ValueError(f"GAN real Ref reference latents must be normalized: {location}")


def _check_ref_metadata(row, location):
    images = row.get("reference_image_count", row.get("ref_image_count"))
    counts = (images, row.get("reference_video_count"), row.get("reference_audio_count"))
    if any(type(count) is not int for count in counts) or not 1 <= images <= 9 or counts[1:] != (0, 0):
        raise ValueError(f"GAN real Ref metadata requires image-only reference counts: {location}")
    if "references" in row:
        references = row["references"]
        if not isinstance(references, list) or len(references) != images or any(
            not isinstance(reference, dict) or str(reference.get("kind", reference.get("type", ""))).lower() != "image"
            or _has_value(reference.get("audio_latents")) or reference.get("include_embedded_audio", False)
            for reference in references
        ):
            raise ValueError(f"GAN real Ref metadata requires image-only references: {location}")
    for key in ("keyframe_anchors", "keyframe_latents", "negative", "negative_conditions"):
        if _has_value(row.get(key)):
            raise ValueError(f"GAN real Ref metadata contains unexpected {key}: {location}")
    for key in ("cache_fingerprint", "latent_fingerprint"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"GAN real Ref metadata requires {key}: {location}")
    if row.get("source_id") is None or row.get("vae_normalized") is not True:
        raise ValueError(f"GAN real Ref metadata requires source_id and VAE-normalized full AV targets: {location}")
    row["reference_image_count"] = images


def load_gan_real_records(real_cache, *, geometry=DEFAULT_GEOMETRY, fps=24, check_files=True, task="t2av"):
    """Return ``(manifest, eligible_rows, sha256)`` without loading tensors.

    Other tasks and different target geometries/frame rates are filtered out.
    Ref2AV may use geometry=None to retain all native geometries; its stream
    selects positives matching the actual rollout geometry at consumption.
    Matching rows must have complete paired AV targets and their own caption and
    condition. The fingerprint binds the entire manifest and resolved row paths;
    it intentionally does not read/hash the potentially huge latent payloads.
    """
    if task not in {"t2av", "ref2av"}:
        raise ValueError("GAN real AV data supports t2av or ref2av")
    geometry = None if geometry is None and task == "ref2av" else tuple(geometry)
    if geometry is not None and (len(geometry) != 3 or any(type(value) is not int or value <= 0 for value in geometry)):
        raise ValueError("GAN target geometry must contain three positive integers.")
    if fps != 24 or geometry is not None and (geometry[0] != 124 or geometry[1] % 32 or geometry[2] % 32):
        raise ValueError("H3 GAN real data requires 124 frames, fps=24 and 32-aligned geometry.")
    manifest = Path(real_cache).expanduser().resolve()
    if manifest.is_dir():
        manifest /= "metadata.jsonl"
    if manifest.suffix.lower() != ".jsonl" or not manifest.is_file():
        raise FileNotFoundError(f"GAN real_cache requires a metadata.jsonl file or cache directory: {manifest}")
    raw = manifest.read_bytes()
    rows = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        location = f"{manifest}:{number}"
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"GAN metadata requires JSON objects: {location}")
        accepted_tasks = {"ref2av", "ref2va"} if task == "ref2av" else {"t2av"}
        if str(row.get("task", "")).lower() not in accepted_tasks:
            continue
        try:
            candidate = tuple(_dimension(row.get(key, row.get(fallback, 0))) for key, fallback in (("target_num_frames", "num_frames"), ("target_height", "height"), ("target_width", "width")))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid GAN target geometry: {location}") from error
        if geometry is not None and candidate != geometry:
            continue
        frames, height, width = candidate
        if frames != 124:
            continue
        if min(height, width) <= 0 or height % 32 or width % 32:
            raise ValueError(f"GAN real Ref target geometry must be positive and 32-aligned: {location}")
        if "fps" not in row:
            raise ValueError(f"GAN real metadata must explicitly declare fps=24: {location}")
        if float(row["fps"]) != fps:
            continue
        row = dict(row)
        row.update(task=task, target_num_frames=frames, target_height=height, target_width=width, fps=fps)
        (_check_ref_metadata if task == "ref2av" else _check_t2av_condition)(row, location)
        if _has_value(row.get("negative_condition_path")):
            raise ValueError(f"GAN real {task.upper()} row contains unexpected negative_condition_path: {location}")
        caption = next((row[key] for key in ("caption", "prompt", "text") if isinstance(row.get(key), str) and row[key].strip()), None)
        if caption is None:
            raise ValueError(f"GAN real data requires its own caption: {location}")
        row["caption"] = caption
        if "vae_normalized" in row and row["vae_normalized"] is not True:
            raise ValueError(f"GAN real AV latents must be VAE-normalized: {location}")
        for key, expected in (("video_latent_shape", [24, 37, height // 16, width // 16]), ("audio_latent_shape", [2, 32, 207])):
            if (task == "ref2av" and key not in row) or key in row and row[key] != expected:
                raise ValueError(f"GAN {key} does not match full target geometry {expected}: {location}")
        for key in CACHE_PATH_KEYS:
            value = row.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"GAN real data requires {key}: {location}")
            path = Path(value).expanduser()
            path = (manifest.parent / path).resolve() if not path.is_absolute() else path.resolve()
            if check_files and not path.is_file():
                raise FileNotFoundError(f"GAN {key} points to a missing file at {location}: {path}")
            row[key] = str(path)
        rows.append(row)
    if not rows:
        raise ValueError(f"GAN real_cache has no eligible {task.upper()} rows with target geometry {geometry} at {fps} fps: {manifest}")
    identity = {"schema": "h3_dmd2_gan_real_v1", "manifest_sha256": hashlib.sha256(raw).hexdigest(), "geometry": geometry, "fps": fps, "rows": rows}
    if task == "ref2av":
        if len({row["condition_path"] for row in rows}) != len(rows):
            raise ValueError(f"GAN real Ref metadata repeats condition paths: {manifest}")
        identity.update(schema="h3_ref_dmd2_gan_real_v1", task=task, condition_schema_version=1)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return manifest, rows, fingerprint


def load_image_gan_real_records(real_cache, *, geometry=(2048, 2048), check_files=True):
    """Validate paired T2I conditions and normalized image latents, without torch.

    Positive samples may be teacher-generated. Their caption/condition identity
    must match their target; the student prompt stream may use other captions.
    """
    geometry = tuple(geometry)
    if len(geometry) != 2 or any(type(value) is not int or value <= 0 or value % 32 for value in geometry):
        raise ValueError("H3 Image GAN geometry must contain two positive 32-aligned integers.")
    height, width = geometry
    manifest = Path(real_cache).expanduser().resolve()
    if manifest.is_dir():
        manifest /= "metadata.jsonl"
    if manifest.suffix.lower() != ".jsonl" or not manifest.is_file():
        raise FileNotFoundError(f"Image GAN real_cache requires metadata.jsonl: {manifest}")
    raw = manifest.read_bytes()
    rows = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        location = f"{manifest}:{number}"
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Image GAN metadata requires JSON objects: {location}")
        if row.get("task") != "t2i":
            continue
        if tuple(_dimension(row.get(key, 0)) for key in ("target_height", "target_width")) != geometry:
            continue
        if row.get("num_frames") != 1 or row.get("latent_shape") != [24, 1, height // 16, width // 16]:
            raise ValueError(f"Image GAN latent_shape must match the single-frame target geometry: {location}")
        if row.get("vae_normalized") is not True:
            raise ValueError(f"Image GAN targets must declare vae_normalized=true: {location}")
        for key in ("prompt", "cache_fingerprint", "real_sample_key", "real_cache_fingerprint"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Image GAN requires {key}: {location}")
        for key in ("references", "negative_condition_path", "negative", "negative_conditions"):
            if _has_value(row.get(key)):
                raise ValueError(f"Image GAN T2I data contains unexpected {key}: {location}")
        for key in ("condition_path", "image_latent_path"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Image GAN requires {key}: {location}")
            path = Path(row[key]).expanduser()
            path = (manifest.parent / path).resolve() if not path.is_absolute() else path.resolve()
            if check_files and not path.is_file():
                raise FileNotFoundError(f"Image GAN {key} is missing at {location}: {path}")
            row[key] = str(path)
        rows.append(row)
    if not rows:
        raise ValueError(f"Image GAN real_cache has no eligible T2I rows with geometry {geometry}: {manifest}")
    identity = {"schema": "h3_image_dmd2_gan_real_v1", "manifest_sha256": hashlib.sha256(raw).hexdigest(), "geometry": geometry, "rows": rows}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return manifest, rows, fingerprint


def _deferred_dataset(manifest, rows, *, task="t2av"):
    from lightx2v_train.data.minimax_h3_cache_dataset import LatentDataset, _condition_payload

    class RealDataset(LatentDataset):
        def _index_metadata(self, metadata_path):
            self.samples.extend({"type": "metadata", "row": row, "base_dir": str(metadata_path.parent)} for row in rows)

        def _load_negative_condition(self, negative_condition_path):
            # A neighboring negative cache belongs to other training tasks.
            return None

        def _validate_condition_item(self, item, row, condition_path):
            if not isinstance(item, dict):
                raise ValueError(f"GAN real condition must contain a {task.upper()} mapping: {condition_path}")
            for container in (item, item.get("conditioning", {})):
                if isinstance(container, dict) and any(_has_value(container.get(key)) for key in ("negative", "negative_conditions")):
                    raise ValueError(f"GAN real {task.upper()} condition contains unexpected negative conditioning: {condition_path}")
            for key in ("cache_fingerprint", "source_id"):
                if key in row and str(item.get(key)) != str(row[key]):
                    raise ValueError(f"GAN real condition {key} differs from its paired target: {condition_path}")
            positive = _condition_payload(item)
            (_check_ref_condition if task == "ref2av" else _check_t2av_condition)(positive, condition_path)
            if task == "ref2av":
                import torch

                if item.get("cache_schema_version") != 1 or item.get("cache_stage") != "complete":
                    raise ValueError(f"GAN real Ref requires a complete schema1 condition: {condition_path}")
                if len(positive["references"]) != row["reference_image_count"]:
                    raise ValueError(f"GAN real Ref reference count differs from paired target: {condition_path}")
                for reference in positive["references"]:
                    tensor = reference.get("video_latents")
                    height, width = reference.get("latent_height"), reference.get("latent_width")
                    if (reference.get("num_latent_frames") != 1 or type(height) is not int or type(width) is not int
                            or min(height, width) <= 0 or height % 2 or width % 2 or not torch.is_tensor(tensor)
                            or tensor.shape != (height * width // 4, 96) or not bool(torch.isfinite(tensor).all())):
                        raise ValueError(f"GAN real Ref has invalid normalized image latent geometry: {condition_path}")
            for key in ("target_num_frames", "target_height", "target_width"):
                if (task == "ref2av" and key not in positive) or key in positive and int(positive[key]) != row[key]:
                    raise ValueError(f"GAN real condition {key} differs from its paired target: {condition_path}")
            tags = positive.get("text_token_tags")
            if task == "ref2av":
                embeds = positive.get("prompt_embeds")
                if (not torch.is_tensor(tags) or not torch.is_tensor(embeds) or embeds.ndim != 2
                        or tags.ndim != 1 or len(tags) != len(embeds) or not bool((tags == 0).any())
                        or not bool(((tags == 0) | (tags == 1)).all()) or not bool(torch.isfinite(embeds).all())):
                    raise ValueError(f"GAN real Ref requires its own multimodal prompt and vision token tags: {condition_path}")
            elif tags is not None and not bool((tags == 1).all()):
                raise ValueError(f"GAN real T2AV condition contains non-text token tags: {condition_path}")

    return RealDataset(str(manifest), defer_latent_loading=True)


def _deferred_image_dataset(manifest, rows):
    from lightx2v_train.data.minimax_h3_image_cache_dataset import MiniMaxH3ImageCacheDataset

    class RealImageDataset(MiniMaxH3ImageCacheDataset):
        def __init__(self):
            # Rows were validated/resolved in preflight. Keep latent IO lazy.
            self.samples = rows

        def __getitem__(self, index):
            sample = super().__getitem__(index)
            for container in (sample, sample["conditioning"], sample["conditioning"]["positive"]):
                for key in ("negative", "negative_conditions", "references"):
                    if _has_value(container.get(key)):
                        raise ValueError(f"Image GAN T2I condition contains unexpected {key}: {manifest}")
            sample["meta"].update({key: self.samples[index][key] for key in ("image_latent_path", "real_sample_key", "real_cache_fingerprint", "latent_shape", "vae_normalized")})
            return sample

    return RealImageDataset()


class DMD2GanRealStream:
    """An infinite shuffled stream indexed by absolute critic microbatches."""

    def __init__(self, real_cache, *, seed, rank=0, world_size=1, geometry=DEFAULT_GEOMETRY, fps=24, task="t2av"):
        if type(world_size) is not int or world_size < 1 or type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("GAN real stream requires a valid data-parallel rank/world_size.")
        if task not in {"t2av", "t2i", "ref2av"}:
            raise ValueError("GAN real stream supports t2av, ref2av or t2i")
        self.task = task
        if task == "t2i":
            self.manifest, self.rows, self.metadata_fingerprint = load_image_gan_real_records(real_cache, geometry=geometry)
        else:
            self.manifest, self.rows, self.metadata_fingerprint = load_gan_real_records(real_cache, geometry=geometry, fps=fps, task=task)
        self.geometry = None if geometry is None else tuple(geometry)
        self.fps = None if task == "t2i" else fps
        self._geometry_indices = {}
        if task == "ref2av":
            for index, row in enumerate(self.rows):
                key = tuple(row[name] for name in ("target_num_frames", "target_height", "target_width"))
                self._geometry_indices.setdefault(key, []).append(index)
        self.seed, self.rank, self.world_size = int(seed), rank, world_size
        self._permutations = OrderedDict()
        self._dataset = None

    def recipe(self):
        recipe = {
            "sampler": "h3_image_gan_permuted_critic_stream_v1" if self.task == "t2i" else "h3_gan_permuted_critic_stream_v1",
            "metadata_fingerprint": self.metadata_fingerprint,
            "rows": len(self.rows),
            "geometry": None if self.geometry is None else list(self.geometry),
            "fps": self.fps,
            "seed": self.seed,
            "world_size": self.world_size,
        }
        if self.task == "ref2av":
            recipe.update(
                sampler="h3_ref_gan_permuted_critic_stream_v1", task=self.task,
                condition_schema_version=1, geometry_policy="exact_rollout_match",
                geometries=[list(key) for key in sorted(self._geometry_indices)],
                geometry_rows=[{"geometry": list(key), "rows": len(self._geometry_indices[key])} for key in sorted(self._geometry_indices)],
            )
        return recipe

    def _requested_geometry(self, geometry):
        if geometry is None:
            geometry = self.geometry
        if geometry is None and len(self._geometry_indices) == 1:
            geometry = next(iter(self._geometry_indices))
        if geometry is None:
            raise ValueError("GAN real Ref stream requires the current rollout geometry for native-resolution positives.")
        geometry = tuple(geometry)
        if geometry not in self._geometry_indices:
            raise ValueError(f"GAN real Ref cache has no positive targets matching rollout geometry {geometry}.")
        return geometry

    def index_at(self, consumed_critic_microbatches, *, geometry=None):
        if type(consumed_critic_microbatches) is not int or consumed_critic_microbatches < 0:
            raise ValueError("GAN real cursor must be a non-negative integer critic microbatch count.")
        if self.task == "ref2av":
            geometry = self._requested_geometry(geometry)
            indices = self._geometry_indices[geometry]
        else:
            indices = range(len(self.rows))
        epoch, offset = divmod(consumed_critic_microbatches * self.world_size + self.rank, len(indices))
        cache_key = (geometry, epoch) if self.task == "ref2av" else epoch
        if cache_key not in self._permutations:
            identity = f"{self.seed}:{self.metadata_fingerprint}:{epoch}"
            if self.task == "ref2av":
                identity += f":{geometry}"
            seed = hashlib.sha256(identity.encode()).digest()
            permutation = list(indices)
            random.Random(int.from_bytes(seed, "big")).shuffle(permutation)
            self._permutations[cache_key] = permutation
            if len(self._permutations) > 2:
                self._permutations.popitem(last=False)
        return self._permutations[cache_key][offset]

    def sample(self, consumed_critic_microbatches, *, geometry=None):
        if self._dataset is None:
            if self.task == "t2i":
                self._dataset = _deferred_image_dataset(self.manifest, self.rows)
            else:
                self._dataset = _deferred_dataset(self.manifest, self.rows, task=self.task)
        return self._dataset[self.index_at(consumed_critic_microbatches, geometry=geometry)]

    def load(self, student, consumed_critic_microbatches, *, broadcast=lambda value: value, geometry=None):
        """Load a clean target with its own positive caption and reference condition."""
        geometry = self._requested_geometry(geometry) if self.task == "ref2av" else self.geometry
        sample = self.sample(consumed_critic_microbatches, geometry=geometry)
        generation_shapes = [{"value": list(geometry)}] if self.task == "t2i" else [list(geometry)]
        shape = student.latent_shape(sample, generation_shapes, broadcast)
        condition, negative = student.encode_conditions(sample, "", 1.0, broadcast)
        if negative is not None:
            raise ValueError("GAN real encoding unexpectedly returned negative conditioning.")
        if self.task == "t2av":
            _check_t2av_condition(condition, self.manifest)
        elif self.task == "ref2av":
            _check_ref_condition(condition, self.manifest)
            if tuple(condition.get(name) for name in ("target_num_frames", "target_height", "target_width")) != geometry:
                raise ValueError("GAN real Ref encoded condition differs from the current rollout geometry.")
            row = self.rows[self.index_at(consumed_critic_microbatches, geometry=geometry)]
            _load_ref_targets(sample, row)
        clean = student.extract_turbo_latents(sample, shape)
        return clean, condition


def _load_ref_targets(sample, row):
    """Validate the paired normalized clean AV before capability packing."""
    import torch

    video = torch.load(row["video_latent_path"], map_location="cpu", weights_only=True)
    audio = torch.load(row["audio_latent_path"], map_location="cpu", weights_only=True)
    if (not isinstance(video, dict) or video.get("normalized") is not True
            or video.get("latent_fingerprint") != row["latent_fingerprint"]):
        raise ValueError("GAN real Ref video target normalization/latent_fingerprint differs from its paired metadata.")
    for tensor, expected, label in ((video.get("latents"), row["video_latent_shape"], "video"), (audio, row["audio_latent_shape"], "audio")):
        if (not torch.is_tensor(tensor) or list(tensor.shape) != expected
                or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"GAN real Ref {label} target must be finite normalized full AV with shape {expected}.")
    sample["inputs"].update(video_latents=video, audio_latents=audio)
