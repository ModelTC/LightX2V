"""Independent, restartable real AV stream for the H3 DMD2 GAN objective.

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


def load_gan_real_records(real_cache, *, geometry=DEFAULT_GEOMETRY, fps=24, check_files=True):
    """Return ``(manifest, eligible_rows, sha256)`` without loading tensors.

    Non-T2AV rows and different target geometries/frame rates are filtered out.
    Matching rows must have complete paired AV targets and their own caption and
    condition. The fingerprint binds the entire manifest and resolved row paths;
    it intentionally does not read/hash the potentially huge latent payloads.
    """
    geometry = tuple(geometry)
    if len(geometry) != 3 or any(type(value) is not int or value <= 0 for value in geometry):
        raise ValueError("GAN target geometry must contain three positive integers.")
    frames, height, width = geometry
    if frames != 124 or fps != 24 or height % 32 or width % 32:
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
        if str(row.get("task", "")).lower() != "t2av":
            continue
        try:
            candidate = tuple(_dimension(row.get(key, row.get(fallback, 0))) for key, fallback in (("target_num_frames", "num_frames"), ("target_height", "height"), ("target_width", "width")))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid GAN target geometry: {location}") from error
        if candidate != geometry:
            continue
        if "fps" not in row:
            raise ValueError(f"GAN real metadata must explicitly declare fps=24: {location}")
        if float(row["fps"]) != fps:
            continue
        row = dict(row)
        row.update(task="t2av", target_num_frames=frames, target_height=height, target_width=width, fps=fps)
        _check_t2av_condition(row, location)
        if _has_value(row.get("negative_condition_path")):
            raise ValueError(f"GAN real T2AV row contains unexpected negative_condition_path: {location}")
        caption = next((row[key] for key in ("caption", "prompt", "text") if isinstance(row.get(key), str) and row[key].strip()), None)
        if caption is None:
            raise ValueError(f"GAN real data requires its own caption: {location}")
        row["caption"] = caption
        if "vae_normalized" in row and row["vae_normalized"] is not True:
            raise ValueError(f"GAN real AV latents must be VAE-normalized: {location}")
        for key, expected in (("video_latent_shape", [24, 37, height // 16, width // 16]), ("audio_latent_shape", [2, 32, 207])):
            if key in row and row[key] != expected:
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
        raise ValueError(f"GAN real_cache has no eligible T2AV rows with target geometry {geometry} at {fps} fps: {manifest}")
    identity = {"schema": "h3_dmd2_gan_real_v1", "manifest_sha256": hashlib.sha256(raw).hexdigest(), "geometry": geometry, "fps": fps, "rows": rows}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return manifest, rows, fingerprint


def _deferred_dataset(manifest, rows):
    from lightx2v_train.data.minimax_h3_cache_dataset import LatentDataset, _condition_payload

    class RealDataset(LatentDataset):
        def _index_metadata(self, metadata_path):
            self.samples.extend({"type": "metadata", "row": row, "base_dir": str(metadata_path.parent)} for row in rows)

        def _load_negative_condition(self, negative_condition_path):
            # A neighboring negative cache belongs to other training tasks.
            return None

        def _validate_condition_item(self, item, row, condition_path):
            if not isinstance(item, dict):
                raise ValueError(f"GAN real condition must contain a T2AV mapping: {condition_path}")
            for container in (item, item.get("conditioning", {})):
                if isinstance(container, dict) and any(_has_value(container.get(key)) for key in ("negative", "negative_conditions")):
                    raise ValueError(f"GAN real T2AV condition contains unexpected negative conditioning: {condition_path}")
            for key in ("cache_fingerprint", "source_id"):
                if key in row and str(item.get(key)) != str(row[key]):
                    raise ValueError(f"GAN real condition {key} differs from its paired target: {condition_path}")
            positive = _condition_payload(item)
            _check_t2av_condition(positive, condition_path)
            for key in ("target_num_frames", "target_height", "target_width"):
                if key in positive and int(positive[key]) != row[key]:
                    raise ValueError(f"GAN real condition {key} differs from its paired target: {condition_path}")
            tags = positive.get("text_token_tags")
            if tags is not None and not bool((tags == 1).all()):
                raise ValueError(f"GAN real T2AV condition contains non-text token tags: {condition_path}")

    return RealDataset(str(manifest), defer_latent_loading=True)


class DMD2GanRealStream:
    """An infinite shuffled stream indexed by absolute critic microbatches."""

    def __init__(self, real_cache, *, seed, rank=0, world_size=1, geometry=DEFAULT_GEOMETRY, fps=24):
        if type(world_size) is not int or world_size < 1 or type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("GAN real stream requires a valid data-parallel rank/world_size.")
        self.manifest, self.rows, self.metadata_fingerprint = load_gan_real_records(real_cache, geometry=geometry, fps=fps)
        self.geometry, self.fps = tuple(geometry), fps
        self.seed, self.rank, self.world_size = int(seed), rank, world_size
        self._permutations = OrderedDict()
        self._dataset = None

    def recipe(self):
        return {
            "sampler": "h3_gan_permuted_critic_stream_v1",
            "metadata_fingerprint": self.metadata_fingerprint,
            "rows": len(self.rows),
            "geometry": list(self.geometry),
            "fps": self.fps,
            "seed": self.seed,
            "world_size": self.world_size,
        }

    def index_at(self, consumed_critic_microbatches):
        if type(consumed_critic_microbatches) is not int or consumed_critic_microbatches < 0:
            raise ValueError("GAN real cursor must be a non-negative integer critic microbatch count.")
        epoch, offset = divmod(consumed_critic_microbatches * self.world_size + self.rank, len(self.rows))
        if epoch not in self._permutations:
            seed = hashlib.sha256(f"{self.seed}:{self.metadata_fingerprint}:{epoch}".encode()).digest()
            permutation = list(range(len(self.rows)))
            random.Random(int.from_bytes(seed, "big")).shuffle(permutation)
            self._permutations[epoch] = permutation
            if len(self._permutations) > 2:
                self._permutations.popitem(last=False)
        return self._permutations[epoch][offset]

    def sample(self, consumed_critic_microbatches):
        if self._dataset is None:
            self._dataset = _deferred_dataset(self.manifest, self.rows)
        return self._dataset[self.index_at(consumed_critic_microbatches)]

    def load(self, student, consumed_critic_microbatches, *, broadcast=lambda value: value):
        """Load clean paired AV and encode its own positive caption condition."""
        sample = self.sample(consumed_critic_microbatches)
        shape = student.latent_shape(sample, [list(self.geometry)], broadcast)
        condition, negative = student.encode_conditions(sample, "", 1.0, broadcast)
        if negative is not None:
            raise ValueError("GAN real T2AV encoding unexpectedly returned negative conditioning.")
        _check_t2av_condition(condition, self.manifest)
        clean = student.extract_turbo_latents(sample, shape)
        return clean, condition
