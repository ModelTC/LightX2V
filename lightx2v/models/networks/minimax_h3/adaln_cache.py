"""Identify, validate, and load persistent MiniMax-H3 AdaLN caches.

Offline generation lives in ``tools/cache_minimax_h3_adaln/builder.py`` so the
inference path does not carry checkpoint-building concerns.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import SafetensorError, safe_open

from lightx2v.models.networks.minimax_h3.adaln_cache_guide import ADALN_CACHE_GUIDE
from lightx2v.models.networks.minimax_h3.packing import (
    CONDITION_AUDIO_TIMESTEP,
    KEYFRAME_NOISE_AUG,
)
from lightx2v.models.schedulers.minimax_h3.scheduler import _make_schedule


def validate_adaln_cache_config(config) -> None:
    """Reject modes whose AdaLN result cannot be represented by this cache."""
    if not config.get("use_adaln_cache", False):
        return
    cache_dir = config.get("adaln_cache_dir")
    if cache_dir is None or not str(cache_dir).strip():
        message = f"\nMINIMAX-H3 ADALN CACHE CONFIGURATION ERROR\n\nuse_adaln_cache=true, but adaln_cache_dir is missing or empty.\n\n{ADALN_CACHE_GUIDE}"
        logger.error(message)
        raise ValueError(message)
    if config.get("dummy_model", False):
        raise NotImplementedError("Persistent MiniMax-H3 AdaLN cache does not support dummy_model")


def _cache_root(config) -> Path:
    return Path(config["adaln_cache_dir"]).expanduser().resolve()


def _selected_profiles(config) -> list[str]:
    model_variant = config["model_variant"]
    if model_variant == "ref2av":
        # Ref2AV always has visual reference rows and may additionally have
        # frozen audio rows. It uses transformer_ref, so its cache must remain
        # separate from every base-transformer task.
        return ["ref2av_video", "ref2av_video_audio"]
    if model_variant == "fl2av":
        # All base-transformer tasks share this pair, so one FL2AV cache also
        # serves T2AV, I2AV, and L2AV without support_tasks-dependent paths.
        return ["t2av", "conditioned"]
    raise ValueError(f"No persistent AdaLN cache profile is available for model_variant: {model_variant!r}")


def _float32_bits(values) -> list[int]:
    # Keep timestep keys bit-exact across JSON serialization and later runs.
    return torch.tensor(list(values), dtype=torch.float32).view(torch.int32).tolist()


def _timesteps_from_bits(bits: list[int], device="cpu") -> torch.Tensor:
    return torch.tensor(bits, dtype=torch.int32).view(torch.float32).to(device)


def _cache_entries(config, profiles: list[str]) -> list[dict]:
    infer_steps = int(config["infer_steps"])
    _, video_timesteps = _make_schedule(
        infer_steps,
        float(config.get("video_flow_shift", 12.0)),
        "cpu",
    )
    _, audio_timesteps = _make_schedule(
        infer_steps,
        float(config.get("audio_flow_shift", 3.0)),
        "cpu",
    )

    entries = []
    for profile in profiles:
        for step, (video_timestep, audio_timestep) in enumerate(zip(video_timesteps.tolist(), audio_timesteps.tolist())):
            values = [video_timestep, audio_timestep]
            if profile in {"conditioned", "ref2av_video", "ref2av_video_audio"}:
                values.append(max(video_timestep, KEYFRAME_NOISE_AUG))
            if profile == "ref2av_video_audio":
                values.append(CONDITION_AUDIO_TIMESTEP)
            unique = torch.unique(torch.tensor(values, dtype=torch.float32), sorted=True)
            entries.append(
                {
                    "name": f"{profile}_step_{step:03d}",
                    "timestep_bits": _float32_bits(unique.tolist()),
                }
            )
    return entries


def is_adaln_cache_key(name: str) -> bool:
    return name.startswith(("time_embedder.", "norm_out.linear.")) or (name.startswith("transformer_blocks.") and ".adaln_proj.linear." in name)


def _checkpoint_files(config) -> list[Path]:
    checkpoint = Path(config["dit_original_ckpt"]).expanduser().resolve()
    files = sorted(checkpoint.glob("*.safetensors")) if checkpoint.is_dir() else [checkpoint]
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"MiniMax-H3 safetensors checkpoint not found: {checkpoint}")
    return files


def _file_signature(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=32)
def _file_sha256(signature):
    with open(signature[0], "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@lru_cache(maxsize=8)
def _base_modulation_sha256(signatures, num_layers):
    """Hash actual cached-weight bytes, streaming without loading the DiT."""
    expected = {f"time_embedder.linear_{index}.{kind}" for index in (1, 2) for kind in ("weight", "bias")}
    expected.update(f"norm_out.linear.{kind}" for kind in ("weight", "bias"))
    expected.update(f"transformer_blocks.{index}.adaln_proj.linear.{kind}" for index in range(num_layers) for kind in ("weight", "bias"))
    found = set()
    digest = hashlib.sha256()
    for signature in signatures:
        with open(signature[0], "rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_size))
            for name in sorted(expected.intersection(header)):
                if name in found:
                    raise ValueError(f"Duplicate MiniMax-H3 modulation tensor: {name}")
                found.add(name)
                entry = header[name]
                digest.update(json.dumps([name, entry["dtype"], entry["shape"]], separators=(",", ":")).encode())
                start, end = entry["data_offsets"]
                handle.seek(8 + header_size + start)
                remaining = end - start
                while remaining:
                    data = handle.read(min(remaining, 8 * 1024 * 1024))
                    if not data:
                        raise ValueError(f"Truncated MiniMax-H3 tensor: {name}")
                    digest.update(data)
                    remaining -= len(data)
    if found != expected:
        raise ValueError(f"Missing MiniMax-H3 modulation tensors: {sorted(expected - found)[:4]}")
    return digest.hexdigest()


def _weight_identity(config):
    files = _checkpoint_files(config)
    identity = {
        "base_checkpoint": str(Path(config["dit_original_ckpt"]).expanduser().resolve()),
        "base_modulation_sha256": _base_modulation_sha256(tuple(_file_signature(path) for path in files), int(config.get("num_layers", 50))),
    }
    if config.get("vdn_checkpoint"):
        from lightx2v.models.networks.minimax_h3.weights.vdn import vdn_adapter_paths

        checkpoint = Path(config["vdn_checkpoint"]).expanduser().resolve()
        identity["vdn"] = {
            "checkpoint": str(checkpoint),
            "spec_sha256": _file_sha256(_file_signature(checkpoint / "model_spec.json")),
            "merge": "default_then_turbo_fp32_delta_cast_then_add_v1",
            "adapters": [{"name": name, "sha256": _file_sha256(_file_signature(path))} for name, path in vdn_adapter_paths(config)],
        }
    if config.get("lora_configs"):
        identity["loras"] = []
        for adapter in config["lora_configs"]:
            path = Path(adapter["path"]).expanduser().resolve()
            # Modulation-changing generic LoRAs cannot use unadapted cache tables.
            with safe_open(path, framework="pt", device="cpu") as source:
                if any(part in key for key in source.keys() for part in ("adaln_proj", "norm_out", "time_embedder")):
                    raise NotImplementedError("AdaLN cache for modulation-changing generic LoRAs is unsupported; use the VDN artifact path or disable the cache")
            identity["loras"].append({"path": str(path), "sha256": _file_sha256(_file_signature(path)), "strength": float(adapter.get("strength", 1.0)), "alpha": adapter.get("alpha")})
    return identity


def _build_spec(config) -> dict:
    if not config.get("use_adaln_cache", False):
        raise ValueError("Building or loading an AdaLN cache requires use_adaln_cache=true")
    validate_adaln_cache_config(config)
    profiles = _selected_profiles(config)
    return {
        "format_version": 2,
        "weight_identity": _weight_identity(config),
        "infer_steps": int(config["infer_steps"]),
        "video_flow_shift": float(config.get("video_flow_shift", 12.0)),
        "audio_flow_shift": float(config.get("audio_flow_shift", 3.0)),
        "num_layers": int(config.get("num_layers", 50)),
        "hidden_size": int(config.get("hidden_size", 5376)),
        "freq_dim": int(config.get("freq_dim", 256)),
        "entries": _cache_entries(config, profiles),
    }


def _cache_path(config, spec) -> Path:
    identity = hashlib.sha256(json.dumps(spec["weight_identity"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    name = f"{config['model_variant']}_{spec['infer_steps']:02d}steps_shift_{spec['video_flow_shift']}_{spec['audio_flow_shift']}_{identity}"
    return _cache_root(config) / "minimax_h3" / name


def _expected_table_shape(spec: dict, entry: dict) -> tuple[int, int]:
    return len(entry["timestep_bits"]) * 3, 6 * spec["hidden_size"]


def _expected_norm_out_shape(spec: dict, entry: dict) -> tuple[int, int]:
    return len(entry["timestep_bits"]), 2 * spec["hidden_size"]


def _norm_out_key(entry: dict) -> str:
    return f"norm_out.{entry['name']}"


def _block_key(block_index: int, entry: dict) -> str:
    return f"block_{block_index:03d}.{entry['name']}"


def _validate_cache(cache_path: Path, spec: dict) -> bool:
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest != spec:
            return False
        expected_keys = {_norm_out_key(entry) for entry in spec["entries"]}
        expected_keys.update(_block_key(block_index, entry) for block_index in range(spec["num_layers"]) for entry in spec["entries"])
        with safe_open(cache_path / "adaln_cache.safetensors", framework="pt", device="cpu") as source:
            if set(source.keys()) != expected_keys:
                return False
            for entry in spec["entries"]:
                tensor = source.get_slice(_norm_out_key(entry))
                if tuple(tensor.get_shape()) != _expected_norm_out_shape(spec, entry) or str(tensor.get_dtype()) != "BF16":
                    return False
            for block_index in range(spec["num_layers"]):
                for entry in spec["entries"]:
                    tensor = source.get_slice(_block_key(block_index, entry))
                    if tuple(tensor.get_shape()) != _expected_table_shape(spec, entry) or str(tensor.get_dtype()) != "BF16":
                        return False
    except (KeyError, OSError, RuntimeError, SafetensorError, TypeError, ValueError):
        return False
    return True


def load_persistent_adaln_cache(
    config,
    device,
) -> tuple[
    dict[tuple[float, ...], list[torch.Tensor]],
    dict[tuple[float, ...], torch.Tensor],
]:
    """Load cached block AdaLN and final-norm modulation onto the device."""
    # Hash weights once on rank 0; broadcast errors too so peers cannot hang.
    if dist.is_initialized() and dist.get_world_size() > 1:
        payload = [None]
        if dist.get_rank() == 0:
            try:
                spec = _build_spec(config)
                payload[0] = {"spec": spec, "path": str(_cache_path(config, spec))}
            except Exception as error:
                payload[0] = {"error": str(error)}
        dist.broadcast_object_list(payload, src=0)
        if "error" in payload[0]:
            raise RuntimeError(f"MiniMax-H3 AdaLN cache identity failed: {payload[0]['error']}")
        spec, cache_path = payload[0]["spec"], Path(payload[0]["path"])
    else:
        spec = _build_spec(config)
        cache_path = _cache_path(config, spec)
    if not _validate_cache(cache_path, spec):
        message = (
            "\nMINIMAX-H3 ADALN CACHE LOAD ERROR\n\n"
            "AdaLN cache not found.\n\n"
            "Inference config:\n"
            f"  Cache root (adaln_cache_dir): {_cache_root(config)}\n"
            f"  infer_steps: {spec['infer_steps']}\n"
            f"  video_flow_shift: {spec['video_flow_shift']}\n"
            f"  audio_flow_shift: {spec['audio_flow_shift']}\n\n"
            "Cache files not found:\n"
            f"  {cache_path / 'manifest.json'}\n"
            f"  {cache_path / 'adaln_cache.safetensors'}\n\n"
            f"{ADALN_CACHE_GUIDE}"
        )
        logger.error(message)
        raise FileNotFoundError(message)

    logger.info("========== Loading MiniMax-H3 AdaLN cache from {} ==========", cache_path)
    keys = [tuple(_timesteps_from_bits(entry["timestep_bits"]).tolist()) for entry in spec["entries"]]
    cache = {key: [None] * spec["num_layers"] for key in keys}
    norm_out_cache = {}
    with safe_open(cache_path / "adaln_cache.safetensors", framework="pt", device=str(device)) as source:
        for entry, key in zip(spec["entries"], keys):
            norm_out_cache[key] = source.get_tensor(_norm_out_key(entry))
        for block_index in range(spec["num_layers"]):
            for entry, key in zip(spec["entries"], keys):
                cache[key][block_index] = source.get_tensor(_block_key(block_index, entry))
    logger.success("========== MiniMax-H3 AdaLN cache loaded from {} ==========", cache_path)
    return cache, norm_out_cache
