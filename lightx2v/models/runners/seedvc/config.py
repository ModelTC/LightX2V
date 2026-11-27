"""Read original Seed-VC YAML metadata without importing the GPU runtime."""

from pathlib import Path

import yaml

from lightx2v.utils.set_config import merge_config

VARIANTS = {
    "svc": ("DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"),
    "vc": ("DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"),
}


def load_config(deployment_config):
    deployment = dict(deployment_config)
    deployment["task"] = deployment.get("task") or "a2a"
    variant = deployment.get("model_variant", "svc")
    if variant == "svc_lora":
        variant = "svc"
        deployment["model_variant"] = variant
        deployment["adapter_path"] = deployment.get("adapter_path") or "adapter.pth"
    if variant not in VARIANTS:
        raise ValueError("Supported Seed-VC variants: svc (V1 44.1k F0), svc_lora, vc (V1 22.05k)")
    root = Path(deployment["model_path"]).expanduser().resolve()
    checkpoint, metadata = VARIANTS[variant]

    def resolve(name, default):
        path = Path(deployment.get(name) or default).expanduser()
        return str((root / path).resolve())

    config_path = resolve("seedvc_config", metadata)
    architecture = yaml.safe_load(Path(config_path).read_text())
    config = merge_config({"architecture": architecture}, deployment)
    config["model_variant"] = variant
    config["dit_original_ckpt"] = resolve("dit_original_ckpt", checkpoint)
    config["vocoder_path"] = resolve("vocoder_path", "vocoder")
    config["seedvc_config"] = config_path
    if config.get("adapter_path"):
        if variant != "svc":
            raise ValueError("Local LoRA adapters require the V1 F0 singing model")
        config["adapter_path"] = resolve("adapter_path", "")
    p = architecture["model_params"]
    d, lr = p["DiT"], p["length_regulator"]
    for name in ("time_as_token", "style_as_token", "uvit_skip_connection"):
        d.setdefault(name, False)
    expected_sr = 44100 if variant == "svc" else 22050
    if architecture["preprocess_params"]["sr"] != expected_sr or lr["f0_condition"] != (variant == "svc"):
        raise ValueError("Checkpoint configuration does not match the selected variant")
    if lr["is_discrete"] or lr["vector_quantize"] or lr["n_codebooks"] != 1 or not lr["sampling_ratios"] or d.get("zero_prompt_speech_token", False):
        raise ValueError("This integration requires the released continuous-content V1 configuration")
    for key in ("cpu_offload", "use_compile", "warmup"):
        config.setdefault(key, False)
    warmup_path = Path(config.get("warmup_sample_path") or Path(__file__).resolve().parents[5] / "lightx2v_examples/a2a/seedvc" / f"{variant}.json")
    if not warmup_path.is_absolute():
        warmup_path = Path(deployment["config_json"]).resolve().parent / warmup_path
    config["warmup_sample_path"] = str(warmup_path.resolve())
    config.setdefault("offload_granularity", "block")
    if config["cpu_offload"] and config["offload_granularity"] != "block":
        raise ValueError("Seed-VC CPU offload uses block granularity")
    for key in ("lazy_load", "dit_quantized", "unload_modules", "enable_reuse", "shared_cpu_weights", "cpu_offload_numa", "cpu_offload_block_contiguous"):
        if config.get(key, False):
            raise ValueError(f"Seed-VC does not support {key}")
    if config.get("lora_configs") or config.get("lora_path") or config.get("disagg_mode"):
        raise ValueError("Use adapter_path for a fixed Seed-VC adapter; dynamic LoRA/disaggregated serving are unsupported")
    if config.get("feature_caching", "NoCaching") != "NoCaching":
        raise ValueError("Seed-VC does not support approximate feature caching")
    parallel = config.get("parallel", {})
    tp = parallel.get("tensor_p_size", 1)
    if tp not in (1, 2) or any(parallel.get(key, 1) != 1 for key in ("cfg_p_size", "seq_p_size", "pp_size")):
        raise ValueError("Seed-VC supports one device or tensor_p_size=2")
    if d["num_heads"] % tp:
        raise ValueError("Attention heads must be divisible by tensor_p_size")
    return config
