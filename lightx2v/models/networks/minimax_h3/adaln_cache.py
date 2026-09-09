"""Identify, validate, and load persistent MiniMax-H3 AdaLN caches.

Offline generation lives in ``tools/cache_minimax_h3_adaln/builder.py`` so the
inference path does not carry checkpoint-building concerns.
"""

import json
from pathlib import Path

import torch
from loguru import logger
from safetensors import SafetensorError, safe_open

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
        separator = "=" * 88
        message = (
            f"\n{separator}\n"
            "MINIMAX-H3 ADALN CACHE CONFIGURATION ERROR\n"
            "use_adaln_cache=true, but adaln_cache_dir is missing or empty.\n"
            "\nACTION REQUIRED\n"
            "Add the cache root to the inference JSON config:\n"
            '  "use_adaln_cache": true,\n'
            '  "adaln_cache_dir": "~/.cache/lightx2v/adaln"\n'
            "adaln_cache_dir can be any custom cache root; the path above is the recommended default.\n"
            "\nBefore inference, set lightx2v_path, model_path, --config_json, and "
            "--task in tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh.\n"
            "Then generate the cache from the repository root with:\n"
            "  bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh\n"
            "Set --task fl2av in that script for t2av/i2av/l2av/fl2av, or "
            "--task ref2av for ref2av. The generation script and inference must "
            f"use the same JSON config.\n{separator}"
        )
        logger.error(message)
        raise ValueError(message)
    if config.get("dummy_model", False):
        raise NotImplementedError("Persistent MiniMax-H3 AdaLN cache does not support dummy_model")


def _cache_root(config) -> Path:
    return Path(config["adaln_cache_dir"]).expanduser().resolve()


def _selected_profiles(config) -> list[str]:
    task = str(config.get("task", ""))
    if task == "ref2av":
        # Ref2AV always has visual reference rows and may additionally have
        # frozen audio rows. It uses transformer_ref, so its cache must remain
        # separate from every base-transformer task.
        return ["ref2av_video", "ref2av_video_audio"]
    if task in {"t2av", "i2av", "l2av", "fl2av"}:
        # All base-transformer tasks share this pair, so one FL2AV cache also
        # serves T2AV, I2AV, and L2AV without support_tasks-dependent paths.
        return ["t2av", "conditioned"]
    raise ValueError(f"No persistent AdaLN cache profile is available for task: {task!r}")


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


def _build_spec(config) -> dict:
    if not config.get("use_adaln_cache", False):
        raise ValueError("Building or loading an AdaLN cache requires use_adaln_cache=true")
    validate_adaln_cache_config(config)
    profiles = _selected_profiles(config)
    return {
        "num_layers": int(config.get("num_layers", 50)),
        "hidden_size": int(config.get("hidden_size", 5376)),
        "freq_dim": int(config.get("freq_dim", 256)),
        "entries": _cache_entries(config, profiles),
    }


def _cache_path(config) -> Path:
    cache_name = "ref2av" if config["task"] == "ref2av" else "fl2av"
    infer_steps = int(config["infer_steps"])
    return _cache_root(config) / "minimax_h3" / f"{cache_name}_{infer_steps:02d}steps"


def _expected_table_shape(spec: dict, entry: dict) -> tuple[int, int]:
    return len(entry["timestep_bits"]) * 3, 6 * spec["hidden_size"]


def _expected_norm_out_shape(spec: dict, entry: dict) -> tuple[int, int]:
    return len(entry["timestep_bits"]), 2 * spec["hidden_size"]


def _validate_cache(cache_path: Path, spec: dict) -> bool:
    manifest_path = cache_path / "manifest.json"
    if not (cache_path / "READY").is_file() or not manifest_path.is_file():
        return False
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest != spec:
            return False
        expected_names = {entry["name"] for entry in spec["entries"]}
        with safe_open(cache_path / "norm_out.safetensors", framework="pt", device="cpu") as source:
            if set(source.keys()) != expected_names:
                return False
            for entry in spec["entries"]:
                tensor = source.get_slice(entry["name"])
                if tuple(tensor.get_shape()) != _expected_norm_out_shape(spec, entry) or str(tensor.get_dtype()) != "BF16":
                    return False
        for block_index in range(spec["num_layers"]):
            block_path = cache_path / f"block_{block_index:03d}.safetensors"
            with safe_open(block_path, framework="pt", device="cpu") as source:
                if set(source.keys()) != expected_names:
                    return False
                for entry in spec["entries"]:
                    tensor = source.get_slice(entry["name"])
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
    spec = _build_spec(config)
    cache_path = _cache_path(config)
    if not _validate_cache(cache_path, spec):
        raise FileNotFoundError(f"MiniMax-H3 AdaLN cache is missing or invalid: {cache_path}. Generate it first with tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py.")

    logger.info("========== Loading MiniMax-H3 AdaLN cache from {} ==========", cache_path)
    keys = [tuple(_timesteps_from_bits(entry["timestep_bits"]).tolist()) for entry in spec["entries"]]
    cache = {key: [None] * spec["num_layers"] for key in keys}
    norm_out_cache = {}
    with safe_open(cache_path / "norm_out.safetensors", framework="pt", device=str(device)) as source:
        for entry, key in zip(spec["entries"], keys):
            norm_out_cache[key] = source.get_tensor(entry["name"])
    for block_index in range(spec["num_layers"]):
        block_path = cache_path / f"block_{block_index:03d}.safetensors"
        with safe_open(block_path, framework="pt", device=str(device)) as source:
            for entry, key in zip(spec["entries"], keys):
                cache[key][block_index] = source.get_tensor(entry["name"])
    logger.success("========== MiniMax-H3 AdaLN cache loaded from {} ==========", cache_path)
    return cache, norm_out_cache
