"""Identify, validate, and load persistent MiniMax-H3 AdaLN caches.

Offline generation lives in ``tools/cache_minimax_h3_adaln/builder.py`` so the
inference path does not carry checkpoint-building concerns.
"""

import hashlib
import json
from importlib import metadata
from pathlib import Path

import torch
from loguru import logger
from safetensors import SafetensorError, safe_open

from lightx2v.models.networks.minimax_h3.adaln_cache_guide import ADALN_CACHE_GUIDE
from lightx2v.models.networks.minimax_h3.packing import (
    CONDITION_AUDIO_TIMESTEP,
    KEYFRAME_NOISE_AUG,
)
from lightx2v.models.schedulers.minimax_h3.scheduler import _make_schedule


def _projection_contract(config):
    """Identify the actual modulation arithmetic and weight files."""
    quantized = bool(config.get("dit_quantized", False))
    scheme = config.get("dit_quant_scheme", "Default") if quantized else "Default"
    if scheme not in ("Default", "fp8-sgl", "fp8-f16-accum", "fp8-f32-accum"):
        raise NotImplementedError(f"H3 AdaLN cache does not support {scheme!r}")
    checkpoint = Path(config["dit_quantized_ckpt"] if quantized else config["dit_original_ckpt"]).expanduser().resolve()
    files = sorted(checkpoint.glob("*.safetensors")) if checkpoint.is_dir() else [checkpoint]
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"H3 AdaLN checkpoint not found: {checkpoint}")
    indices = sorted(checkpoint.glob("*.safetensors.index.json")) if checkpoint.is_dir() else []
    checkpoint_metadata = {}
    if quantized:
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                checkpoint_metadata[path.name] = handle.metadata() or {}
    try:
        sgl_version = metadata.version("sgl-kernel") if quantized else None
    except metadata.PackageNotFoundError:
        sgl_version = "unavailable"
    return {
        "version": 1,
        # Both accumulation modes leave AdaLN on the same SGL path.
        "scheme": "fp8-f16-accum" if scheme == "fp8-f32-accum" else scheme,
        **({"checkpoint_layout": "native_h3" if quantized else "streamrender_release"} if config["model_cls"] == "streamrender_h3" else {}),
        # AdaLN is intentionally absent from H3_PROJECTION_SUFFIXES.
        # Its FP8-F16 profile therefore uses this same SGL per-token backend.
        "adaln_backend": "sgl_fp8_scaled_mm" if quantized else "torch.addmm",
        "activation_quantization": "sgl_per_token_quant_fp8" if quantized else None,
        "activation_qmax": 448.0 if quantized else None,
        "time_mlp_dtype": "fp32",
        "modulation_dtype": "bf16",
        "torch_version": str(torch.__version__),
        "cuda_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "sgl_kernel_version": sgl_version,
        "checkpoint_metadata": checkpoint_metadata,
        # Names/stat metadata survive a verified copy2 SHM cache. The index
        # digest additionally binds shard/tensor mappings without rereading
        # tens of GiB every time a runner validates its small AdaLN cache.
        "checkpoint": [{"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in files],
        "index_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in indices},
    }


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
    if config["model_cls"] == "minimax_h3_causal":
        return ["refa2v_condition", "refa2v_media"]
    if config["model_cls"] == "taomate_h3":
        if config.get("adaln_cache_profile") == "taomate_base10":
            if int(config["infer_steps"]) != 9:
                raise ValueError("TaoMate Base10 AdaLN cache requires nine denoising steps")
            return ["taomate_base10_first", "taomate_base10_continue"]
        if config.get("adaln_cache_profile") is not None:
            raise ValueError(f"Unknown TaoMate AdaLN profile: {config['adaln_cache_profile']!r}")
        return ["taomate_stage3"]
    if config["model_cls"] == "streamrender_h3":
        if int(config["infer_steps"]) != 2:
            raise ValueError("StreamRender AdaLN cache requires two denoising steps")
        return ["streamrender"]
    model_variant = config["model_variant"]
    if model_variant == "ref2av":
        # Ref2AV always has visual reference rows and may additionally have
        # frozen audio rows. It uses transformer_ref, so its cache must remain
        # separate from every base-transformer task.
        return ["ref2av_video", "ref2av_video_audio"]
    if model_variant == "fl2av":
        # Visual-only references share the conditioned profile. Reference audio
        # adds a clean-audio timestep; disk-streaming text encoders serve T2AV only.
        profiles = ["t2av", "conditioned"]
        if not config.get("text_encoder_disk_streaming", False):
            profiles.append("ref2av_video_audio")
        return profiles
    raise ValueError(f"No persistent AdaLN cache profile is available for model_variant: {model_variant!r}")


def _float32_bits(values) -> list[int]:
    # Keep timestep keys bit-exact across JSON serialization and later runs.
    return torch.tensor(list(values), dtype=torch.float32).view(torch.int32).tolist()


def _timesteps_from_bits(bits: list[int], device="cpu") -> torch.Tensor:
    return torch.tensor(bits, dtype=torch.int32).view(torch.float32).to(device)


def _cache_entries(config, profiles: list[str]) -> list[dict]:
    if profiles == ["streamrender"]:
        from lightx2v.models.schedulers.streamrender_h3.scheduler import _shifted_timesteps

        # Match pre-inference's FP32 subtraction, including the clean source
        # timestep. Rounded decimal fractions are not substitutes for its bits.
        video = 1.0 - _shifted_timesteps(6)
        audio = 1.0 - _shifted_timesteps(1.5)
        clean = 1.0 - torch.tensor(0.001, dtype=torch.float32)
        return [
            {"name": f"streamrender_step_{step:03d}", "timestep_bits": _float32_bits(torch.unique(torch.stack((v, a, clean)), sorted=True).tolist())} for step, (v, a) in enumerate(zip(video, audio))
        ]
    if profiles == ["taomate_stage3"]:
        from lightx2v.models.schedulers.taomate_h3.scheduler import distilled_sigmas

        video_timesteps = 1.0 - distilled_sigmas(float(config.get("video_flow_shift", 12.0)))[:-1]
        audio_timesteps = 1.0 - distilled_sigmas(float(config.get("audio_flow_shift", 3.0)))[:-1]
        entries = []
        for step, (video, audio) in enumerate(zip(video_timesteps, audio_timesteps)):
            timesteps = torch.unique(torch.stack((video, audio)), sorted=True)
            entries.append({"name": f"taomate_stage3_step_{step:03d}", "timestep_bits": _float32_bits(timesteps.tolist())})
        entries.append({"name": "taomate_stage3_clean", "timestep_bits": _float32_bits((1.0,))})
        return entries

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
            if profile == "refa2v_condition":
                # Prefill has timestep-dependent text and fixed noisy references.
                values = [video_timestep, 0.999]
            elif profile == "refa2v_media":
                # Each media chunk contains noisy video and clean audio.
                values = [video_timestep, 1.0]
            if profile in {"conditioned", "ref2av_video", "ref2av_video_audio"}:
                values.append(max(video_timestep, KEYFRAME_NOISE_AUG))
            if profile in {"ref2av_video_audio", "taomate_base10_continue"}:
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
    spec = {
        "infer_steps": int(config["infer_steps"]),
        "video_flow_shift": float(config.get("video_flow_shift", 12.0)),
        "audio_flow_shift": float(config.get("audio_flow_shift", 3.0)),
        "num_layers": int(config.get("num_layers", 50)),
        "hidden_size": int(config.get("hidden_size", 5376)),
        "freq_dim": int(config.get("freq_dim", 256)),
        "entries": _cache_entries(config, profiles),
    }
    if config["model_cls"] == "minimax_h3_causal":
        # Track the time-MLP precision so caches from the old all-BF16 policy
        # cannot be reused. Checkpoint replacement also invalidates the cache.
        checkpoint = Path(config["dit_original_ckpt"]).expanduser().resolve()
        files = sorted(checkpoint.glob("*.safetensors")) if checkpoint.is_dir() else [checkpoint]
        spec["projection_dtype"] = "fp32"
        spec["checkpoint"] = [{"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in files]
    elif config["model_cls"] in {"taomate_h3", "streamrender_h3"}:
        spec["projection_contract"] = _projection_contract(config)
    return spec


def _cache_path(config) -> Path:
    model = config["model_cls"]
    cache_name = "refa2v" if model == "minimax_h3_causal" else "refv2v" if model == "streamrender_h3" else config.get("adaln_cache_profile") or config["model_variant"]
    infer_steps = int(config["infer_steps"])
    video_flow_shift = float(config.get("video_flow_shift", 12.0))
    audio_flow_shift = float(config.get("audio_flow_shift", 3.0))
    return _cache_root(config) / model / f"{cache_name}_{infer_steps:02d}steps_shift_{video_flow_shift}_{audio_flow_shift}"


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
    spec = _build_spec(config)
    cache_path = _cache_path(config)
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
    with safe_open(cache_path / "adaln_cache.safetensors", framework="pt", device="cpu") as source:
        for entry, key in zip(spec["entries"], keys):
            norm_out_cache[key] = source.get_tensor(_norm_out_key(entry)).to(device)
        for block_index in range(spec["num_layers"]):
            for entry, key in zip(spec["entries"], keys):
                cache[key][block_index] = source.get_tensor(_block_key(block_index, entry)).to(device)
    logger.success("========== MiniMax-H3 AdaLN cache loaded from {} ==========", cache_path)
    return cache, norm_out_cache
