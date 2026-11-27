"""Resolve checkpoint architecture without importing torch or allocating CUDA."""

import json
from pathlib import Path

from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    deployment = dict(deployment_config)
    if deployment.get("task", None) is None:
        deployment["task"] = "t2a"
    root = Path(deployment["model_path"]).expanduser().resolve()
    metadata = json.loads((root / "config.json").read_text())
    if metadata.get("model_type") != "yue2" or metadata.get("latent_dim") != 64:
        raise ValueError("Expected the YuE2 AR-NAR checkpoint (latent_dim=64)")
    config = merge_config(metadata, deployment)
    vae_path = config.get("audio_vae_original_ckpt", None)
    vae = Path(root.parent / "YuE2-Vae" if vae_path is None else vae_path).expanduser()
    if not vae.is_absolute():
        vae = Path(deployment.get("config_json", ".")).resolve().parent / vae
    if not (vae / "config.json").is_file():
        raise FileNotFoundError(f"YuE2 VAE directory not found: {vae}")
    config["audio_vae_original_ckpt"] = str(vae.resolve())
    config.setdefault("cpu_offload", False)
    config.setdefault("offload_granularity", "block")
    config.setdefault("use_compile", False)
    config.setdefault("warmup", False)
    for flag in ("lazy_load", "unload_modules", "dit_quantized", "enable_reuse", "shared_cpu_weights", "cpu_offload_numa"):
        if config.get(flag, False):
            raise ValueError(f"YuE2 does not support {flag}")
    if config.get("lora_configs", []) or config.get("lora_path", None) or config.get("disagg_mode", None):
        raise ValueError("YuE2 does not support LoRA or disaggregated serving")
    if config.get("feature_caching", "NoCaching") != "NoCaching":
        raise ValueError("YuE2 does not support approximate feature caching")
    if config["cpu_offload"] and config["offload_granularity"] != "block":
        raise ValueError("YuE2 CPU offload uses block granularity")
    parallel = config.get("parallel", {})
    tp = parallel.get("tensor_p_size", 1)
    if tp not in (1, 2) or any(parallel.get(key, 1) != 1 for key in ("cfg_p_size", "seq_p_size", "pp_size")):
        raise ValueError("YuE2 supports a single device or tensor_p_size=2")
    for key in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
        if config[key] % tp:
            raise ValueError(f"{key} is not divisible by tensor_p_size")
    return config
