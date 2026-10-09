"""Paired, condition-aligned real/teacher targets for H3 Ref2AV DMAD.

The dataset extends the existing condition cache reader and uses the same
reference-count/orientation/cost sampler. Each joined metadata row requires
condition_path, real_latent_path, teacher_latent_path, cache_fingerprint, and
target_height/target_width/target_num_frames. See minimax_h3_dmad_manifest for
the metadata-only join CLI. A condition-only cache is not a DMAD dataset.

Each target .pt contains a plain dictionary::

    {
        "normalized": True,  # already normalized by the H3 VAE statistics
        "cache_fingerprint": "exact-condition-fingerprint",
        "target_height": 768, "target_width": 1344,
        "target_num_frames": 124,
        "video": Tensor[V, 96], "audio": Tensor[A, 32],
    }

video_latents/audio_latents are accepted aliases. A leading singleton batch
is accepted. Canonical video [24,F,H/16,W/16] is patchified with (1,2,2), and
canonical stereo audio [2,T,32] (or the VAE-native [2,32,T]) is flattened in
channel-major order, matching the H3 encoder. All targets must already be normalized; raw encoder outputs
are never silently treated as normalized. Optional condition_path/source IDs
are cross-checked; the cache_fingerprint and geometry are always required.

Samples retain the original inputs/conditioning/meta contract and add
dmad_real and dmad_teacher, each {video: Tensor[V,96], audio: Tensor[A,32]}.
Default singleton collation adds one leading batch axis to those tensors.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lightx2v_train.data.minimax_h3_cache_dataset import (
    LatentDataset,
    MiniMaxH3ReferenceCostSampler,
    _build_dataloader,
    _condition_payload,
    _metadata_path,
)
from lightx2v_train.data.minimax_h3_dmad_manifest import (
    require_matching_identity,
    resolve_manifest_path,
    target_geometry,
    validate_manifest,
)
from lightx2v_train.runtime.distributed import get_data_parallel_rank, get_data_parallel_world_size
from lightx2v_train.utils.registry import DATA_REGISTER


def _finite_tensors(value, label):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"DMAD {label} contains non-finite tensors.")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tensors(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tensors(item, f"{label}[{index}]")


def _target_tensor(payload, key):
    alias = f"{key}_latents"
    if key in payload and alias in payload:
        raise ValueError(f"DMAD target must not specify both {key} and {alias}.")
    tensor = payload.get(key, payload.get(alias))
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        raise TypeError(f"DMAD target {key} must be a floating-point tensor.")
    if tensor.layout != torch.strided:
        raise ValueError(f"DMAD target {key} must be a dense strided tensor.")
    _finite_tensors(tensor, f"target.{key}")
    return tensor


def pack_target_payload(payload, row, *, label, base_dir):
    """Validate identity/geometry and return unbatched, normalized fp32 rows."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"DMAD {label} latent payload must be a dictionary.")
    if payload.get("normalized") is not True:
        raise ValueError(f"DMAD {label} target must declare normalized=true.")
    require_matching_identity(row, payload, label)
    geometry = target_geometry(row)
    if target_geometry(payload) != geometry:
        raise ValueError(f"DMAD {label} target geometry does not match its condition.")
    if "condition_path" in payload:
        actual = resolve_manifest_path(payload["condition_path"], base_dir, "condition_path")
        if actual != row["condition_path"]:
            raise ValueError(f"DMAD {label} condition_path does not match its condition.")

    height, width, frames = geometry
    latent_frames = (frames - 5) // 17 * 5 + 2
    latent_height, latent_width = height // 16, width // 16
    video_rows = latent_frames * (height // 32) * (width // 32)
    audio_frames = int(round(frames / 24 * 40))
    video = _target_tensor(payload, "video")
    if video.ndim in (3, 5) and video.shape[0] == 1:
        video = video[0]
    if video.ndim == 4:
        expected = (24, latent_frames, latent_height, latent_width)
        if tuple(video.shape) != expected:
            raise ValueError(f"DMAD {label} canonical video shape must be {expected}, got {tuple(video.shape)}.")
        # Identical axis order to native.minimax_h3.patchify_video_latents,
        # without importing model code into CPU data workers.
        video = video.reshape(24, latent_frames, latent_height // 2, 2, latent_width // 2, 2)
        video = video.permute(1, 2, 4, 0, 3, 5).reshape(-1, 96)
    if tuple(video.shape) != (video_rows, 96):
        raise ValueError(f"DMAD {label} packed video shape must be {(video_rows, 96)}, got {tuple(video.shape)}.")

    audio = _target_tensor(payload, "audio")
    if audio.ndim in (3, 4) and audio.shape[0] == 1:
        audio = audio[0]
    if tuple(audio.shape) == (2, 32, audio_frames):
        audio = audio.transpose(1, 2)
    if tuple(audio.shape) == (2, audio_frames, 32):
        audio = audio.reshape(-1, 32)
    if tuple(audio.shape) != (2 * audio_frames, 32):
        raise ValueError(f"DMAD {label} packed audio shape must be {(2 * audio_frames, 32)}, got {tuple(audio.shape)}.")
    result = {"video": video.to(dtype=torch.float32).contiguous(), "audio": audio.to(dtype=torch.float32).contiguous()}
    # Conversion can overflow otherwise-finite float64 input.
    _finite_tensors(result, label)
    return result


class MiniMaxH3DMADDataset(LatentDataset):
    """A physical condition row and its two exact-identity target latents."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifest_digest = self._manifest_digest()

    def _manifest_digest(self):
        """Hash actual selected physical rows, in their deterministic order.

        Compute after LatentDataset applies max_samples. Manifest locations,
        JSON formatting, and dataset_repeat aliases are not identities; an
        equivalent selected sequence has the same digest even if those change.
        Paths are resolved during indexing. This metadata digest deliberately
        does not claim to hash artifact contents or detect in-place tensor edits.
        """
        digest = hashlib.sha256(b"minimax-h3-dmad-selected-manifest-v1\n")
        for sample in self.samples:
            row = sample["row"]
            height, width, frames = target_geometry(row)
            latent_frames = (frames - 5) // 17 * 5 + 2
            identity = {
                "schema_version": row["dmad_schema_version"],
                "cache_fingerprint": row["cache_fingerprint"],
                "paths": {key: row[key] for key in ("condition_path", "real_latent_path", "teacher_latent_path")},
                "source_ids": {key: str(row[key]) for key in ("source_id", "source_row_uid", "sample_id", "id") if row.get(key) not in (None, "")},
                "target_geometry": [height, width, frames],
                "packed_video_shape": [latent_frames * (height // 32) * (width // 32), 96],
                "packed_audio_shape": [2 * int(round(frames / 24 * 40)), 32],
            }
            digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def _index_path(self, path):
        self._index_metadata(_metadata_path(path))

    def _index_metadata(self, metadata_path):
        existing_paths = {sample["row"]["condition_path"] for sample in self.samples}
        existing_fingerprints = {sample["row"]["cache_fingerprint"] for sample in self.samples}
        for row in validate_manifest(metadata_path):
            if row["condition_path"] in existing_paths or row["cache_fingerprint"] in existing_fingerprints:
                raise ValueError("Duplicate DMAD condition across dataset manifests.")
            existing_paths.add(row["condition_path"])
            existing_fingerprints.add(row["cache_fingerprint"])
            self.samples.append({"type": "metadata", "row": row, "base_dir": str(metadata_path.parent)})

    def _validate_condition_item(self, item, row, condition_path):
        if not isinstance(item, Mapping):
            raise TypeError("DMAD Ref2AV condition cache must contain a dictionary.")
        require_matching_identity(row, item, "condition cache")
        positive = _condition_payload(item)
        if not isinstance(positive, Mapping) or target_geometry(positive) != target_geometry(row):
            raise ValueError("DMAD condition cache geometry does not match metadata.")
        if positive.get("task", "ref2av") not in ("ref2av", "ref2va"):
            raise ValueError("DMAD condition cache must be Ref2AV.")
        _finite_tensors(item, "condition cache")

    def _load_metadata_sample(self, row, base_dir):
        sample = super()._load_metadata_sample(row, base_dir)
        # Includes separately loaded negative conditions as well as positive.
        _finite_tensors(sample["conditioning"], "conditioning")
        for role in ("real", "teacher"):
            path = Path(row[f"{role}_latent_path"])
            payload = torch.load(path, map_location="cpu", weights_only=True)
            sample[f"dmad_{role}"] = pack_target_payload(payload, row, label=role, base_dir=path.parent)
            sample["meta"][f"{role}_latent_path"] = str(path)
        sample["meta"]["cache_fingerprint"] = row["cache_fingerprint"]
        sample["meta"]["dmad_schema_version"] = row["dmad_schema_version"]
        return sample


@DATA_REGISTER("minimax_h3_dmad_dataset")
def build_minimax_h3_dmad_dataset(data_config, train_or_val="train", unconditional_prompt=" ", sample_processor=None):
    if int(data_config.get("batch_size", 1)) != 1:
        raise ValueError("MiniMax-H3 DMAD packed training requires data.train.batch_size=1.")
    dataset = MiniMaxH3DMADDataset(
        data_paths=data_config["data_path"],
        dataset_repeat=data_config.get("dataset_repeat", 1),
        max_samples=data_config.get("max_samples"),
        prompt_column=data_config.get("prompt_column", "caption"),
        prompt_index=data_config.get("prompt_index", 0),
        negative_condition_path=data_config.get("negative_condition_path"),
        # Real/teacher targets are loaded separately, never confuse reference
        # latents or legacy video_latent_path with real supervision.
        defer_latent_loading=True,
    )
    if train_or_val != "train":
        return _build_dataloader(dataset, data_config, train_or_val)
    sampler_config = data_config.get("reference_cost_sampler", {})
    if not isinstance(sampler_config, Mapping):
        raise TypeError("data.train.reference_cost_sampler must be a mapping.")
    sampler = MiniMaxH3ReferenceCostSampler(
        dataset,
        num_replicas=get_data_parallel_world_size(),
        rank=get_data_parallel_rank(),
        seed=sampler_config.get("seed", data_config.get("seed", 0)),
        cost_key=sampler_config.get("cost_key", "packed_sequence_tokens_124"),
        require_compute_cost=sampler_config.get("require_compute_cost", True),
        require_image_only=sampler_config.get("require_image_only", True),
        image_counts=sampler_config.get("image_counts"),
        require_all_image_counts=sampler_config.get("require_all_image_counts", True),
        balance_image_counts=sampler_config.get("balance_image_counts", False),
        balance_orientation=sampler_config.get("balance_orientation", True),
        strict_full_epoch=sampler_config.get("strict_full_epoch", True),
        remainder_policy=sampler_config.get("remainder_policy"),
        batch_mode=sampler_config.get("batch_mode", "cost_local"),
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=data_config.get("num_workers", 0),
        pin_memory=data_config.get("pin_memory", False),
        drop_last=False,
    )
