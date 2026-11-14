"""Atomic, target-only MiniMax-H3 teacher clean latent artifacts.

The manifest is the completion marker. A failed or interrupted export may leave
individual files, but never publishes a manifest without the final clean AV.
This module intentionally has no model/platform imports so its disk contract
can be checked on CPU.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist


def run_rank_zero(action):
    """Run file IO on rank zero, propagating failures before the next collective."""
    distributed = dist.is_available() and dist.is_initialized()
    error = [None]
    result = None
    if not distributed or dist.get_rank() == 0:
        try:
            result = action()
        except Exception as exc:
            if not distributed:
                raise
            error[0] = f"{type(exc).__name__}: {exc}"
    if distributed:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"MiniMax-H3 teacher export failed: {error[0]}")
    return result


def run_collective_action(action):
    """Keep all VAE ranks participating and propagate decode/save errors."""
    if not dist.is_available() or not dist.is_initialized():
        return action()
    result, error = None, None
    try:
        result = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(value is not None for value in errors):
        raise RuntimeError(f"MiniMax-H3 teacher decode failed: {errors}")
    return result


def path_identity(path):
    """Record local identity without reading multi-gigabyte checkpoint weights."""
    path = Path(path).expanduser().absolute()
    stat = path.stat()
    identity = {"path": str(path), "resolved_path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if path.is_dir():
        # Merged checkpoints commonly link text encoders and VAEs from the
        # original model. Follow those links while guarding directory cycles.
        files, visited = [], set()
        for directory, directories, filenames in os.walk(path, followlinks=True):
            resolved = os.path.realpath(directory)
            if resolved in visited:
                directories[:] = []
                continue
            visited.add(resolved)
            directories.sort()
            for filename in sorted(filenames):
                child = Path(directory) / filename
                child_stat = child.stat()
                files.append({"path": str(child.relative_to(path)), "resolved_path": str(child.resolve()), "size": child_stat.st_size, "mtime_ns": child_stat.st_mtime_ns})
        identity["files"] = files
    return identity


def fingerprint_metadata(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def inference_config_fingerprint(config):
    # Runtime config also holds process groups and tensors; only JSON values
    # describe reproducible startup settings.
    serializable = {}
    for key, value in config.items():
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            continue
        serializable[key] = value
    return fingerprint_metadata(serializable)


def merged_checkpoint_identity(model_path, transformer_path):
    candidates = [Path(model_path) / "decfg_merged.json", Path(transformer_path) / "decfg_merged.json", Path(transformer_path).parent / "decfg_merged.json"]
    identities = []
    for path in dict.fromkeys(path.absolute() for path in candidates):
        if path.is_file():
            data = path.read_bytes()
            identities.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "metadata": json.loads(data)})
    return identities


def _atomic_write(path, write):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class H3TeacherFinalWriter:
    """Synchronous writer; callers invoke this class on rank zero only."""

    def __init__(self, directory, metadata, condition):
        self.directory = Path(directory).expanduser().absolute()
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "manifest.json").exists():
            raise FileExistsError(f"Completed teacher export already exists: {self.directory}")
        self.metadata = dict(metadata)
        self.num_steps = int(metadata["num_steps"])
        self.final_written = False
        for key in ("video_sigmas", "audio_sigmas"):
            sigmas = metadata[key]
            if len(sigmas) != self.num_steps + 1 or sigmas[0] != 1.0 or sigmas[-1] != 0.0 or any(a <= b for a, b in zip(sigmas, sigmas[1:])):
                raise ValueError(f"Teacher export requires a complete descending {key} schedule from 1 to 0")
        self._save("condition.pt", condition)

    def _save(self, name, value):
        _atomic_write(self.directory / name, lambda handle: torch.save(value, handle))

    def write_final(self, index, video_latents, audio_latents):
        if index != self.num_steps or self.final_written:
            raise ValueError(f"Expected one final teacher state at step {self.num_steps}, got {index}")
        tensors = {}
        for kind, value in (("video", video_latents), ("audio", audio_latents)):
            value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
            if list(value.shape) != self.metadata[f"{kind}_shape"]:
                raise ValueError(f"Unexpected {kind} teacher latent shape: {list(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite {kind} teacher latents at state {index}")
            tensors[f"{kind}_latents"] = value
        self._save("video_latents.pt", {"normalized": True, "latents": tensors["video_latents"]})
        self._save("audio_latents.pt", tensors["audio_latents"])
        self.final_written = True

    def finish(self):
        if not self.final_written:
            raise RuntimeError("Cannot complete an incomplete teacher export without final AV latents")
        manifest = {
            **self.metadata,
            "schema": "h3_teacher_final_v1",
            "export_mode": "final",
            "normalized": True,
            "dtype": "float32",
            "num_states": 1,
            "state_paths": [],
            "final_state_index": self.num_steps,
            "final_video_sigma": self.metadata["video_sigmas"][-1],
            "final_audio_sigma": self.metadata["audio_sigmas"][-1],
            "condition_path": "condition.pt",
            "video_latents_path": "video_latents.pt",
            "audio_latents_path": "audio_latents.pt",
        }
        encoded = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        _atomic_write(self.directory / "manifest.json", lambda handle: handle.write(encoded))
        return str(self.directory / "manifest.json")
