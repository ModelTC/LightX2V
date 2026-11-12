"""Identify, validate, and load persistent MiniMax-H3 AdaLN caches.

Offline generation lives in ``tools/cache_minimax_h3_adaln/builder.py`` so the
inference path does not carry checkpoint-building concerns.
"""

import json
from pathlib import Path

import torch
from loguru import logger
from safetensors import SafetensorError, safe_open

from lightx2v.models.networks.minimax_h3.adaln_cache_guide import get_adaln_cache_guide
from lightx2v.models.networks.minimax_h3.adaln_profiles import cache_plan


def checkpoint_files(checkpoint) -> list[Path]:
    """Resolve a cache source checkpoint and validate its tensor files."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    files = sorted(checkpoint.glob("*.safetensors")) if checkpoint.is_dir() else [checkpoint]
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"H3 AdaLN checkpoint not found: {checkpoint}")
    return files


def validate_adaln_cache_config(config) -> None:
    """Reject modes whose AdaLN result cannot be represented by this cache."""
    if not config.get("use_adaln_cache", False):
        return
    cache_dir = config.get("adaln_cache_dir")
    if cache_dir is None or not str(cache_dir).strip():
        message = f"\nMINIMAX-H3 ADALN CACHE CONFIGURATION ERROR\n\nuse_adaln_cache=true, but adaln_cache_dir is missing or empty.\n\n{get_adaln_cache_guide(config['model_cls'])}"
        logger.error(message)
        raise ValueError(message)
    if config.get("dummy_model", False):
        raise NotImplementedError("Persistent MiniMax-H3 AdaLN cache does not support dummy_model")


def _cache_root(config) -> Path:
    return Path(config["adaln_cache_dir"]).expanduser().resolve()


def _float32_bits(values) -> list[int]:
    # Keep timestep keys bit-exact across JSON serialization and later runs.
    return torch.tensor(list(values), dtype=torch.float32).view(torch.int32).tolist()


def _timesteps_from_bits(bits: list[int], device="cpu") -> torch.Tensor:
    return torch.tensor(bits, dtype=torch.int32).view(torch.float32).to(device)


def _image_cache_spec(config, plan) -> dict:
    """Image modulation depends on PDD timesteps, without AV flow shifts."""
    return {
        "infer_steps": int(config["infer_steps"]),
        "sigmas": list(plan.sigmas),
        "num_layers": int(config.get("num_layers", 50)),
        "hidden_size": int(config.get("hidden_size", 5376)),
        "freq_dim": int(config.get("freq_dim", 256)),
        "entries": [{"name": name, "timestep_bits": _float32_bits(values.tolist())} for name, values in plan.timesteps.items()],
    }


def _video_cache_spec(config, plan) -> dict:
    """Video modulation follows the separate video and audio flow schedules."""
    return {
        "infer_steps": int(config["infer_steps"]),
        "video_flow_shift": float(config.get("video_flow_shift", 12.0)),
        "audio_flow_shift": float(config.get("audio_flow_shift", 3.0)),
        "num_layers": int(config.get("num_layers", 50)),
        "hidden_size": int(config.get("hidden_size", 5376)),
        "freq_dim": int(config.get("freq_dim", 256)),
        "entries": [{"name": name, "timestep_bits": _float32_bits(values.tolist())} for name, values in plan.timesteps.items()],
    }


def resolve_adaln_cache(config) -> tuple[Path, dict]:
    """Resolve the cache directory and expected manifest from one model plan."""
    if not config.get("use_adaln_cache", False):
        raise ValueError("Building or loading an AdaLN cache requires use_adaln_cache=true")
    validate_adaln_cache_config(config)
    plan = cache_plan(config)
    if config["model_cls"] == "minimax_h3_image":
        spec = _image_cache_spec(config, plan)
    else:
        spec = _video_cache_spec(config, plan)
    cache_path = _cache_root(config) / config["model_cls"] / plan.cache_name
    return cache_path, spec


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
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, SafetensorError):
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
    cache_path, spec = resolve_adaln_cache(config)
    if not _validate_cache(cache_path, spec):
        config_summary = "".join(f"  {key}: {value}\n" for key, value in spec.items() if key != "entries")
        message = (
            "\nMINIMAX-H3 ADALN CACHE LOAD ERROR\n\n"
            "AdaLN cache is missing or does not match the current model, computation, or format.\n\n"
            "Inference config:\n"
            f"  Cache root (adaln_cache_dir): {_cache_root(config)}\n"
            f"{config_summary}\n"
            "Expected cache files:\n"
            f"  {cache_path / 'manifest.json'}\n"
            f"  {cache_path / 'adaln_cache.safetensors'}\n\n"
            f"{get_adaln_cache_guide(config['model_cls'])}"
        )
        logger.error(message)
        raise FileNotFoundError(message)

    logger.info("========== Loading MiniMax-H3 AdaLN cache from {} ==========", cache_path)
    keys = [tuple(_timesteps_from_bits(entry["timestep_bits"]).tolist()) for entry in spec["entries"]]
    cache = {key: [None] * spec["num_layers"] for key in keys}
    norm_out_cache = {}
    with safe_open(cache_path / "adaln_cache.safetensors", framework="pt", device="cpu") as source:
        for entry, key in zip(spec["entries"], keys):
            norm_out_cache[key] = source.get_tensor(_norm_out_key(entry)).to(device)
        for block_index in range(spec["num_layers"]):
            for entry, key in zip(spec["entries"], keys):
                cache[key][block_index] = source.get_tensor(_block_key(block_index, entry)).to(device)
    logger.success("========== MiniMax-H3 AdaLN cache loaded from {} ==========", cache_path)
    return cache, norm_out_cache
