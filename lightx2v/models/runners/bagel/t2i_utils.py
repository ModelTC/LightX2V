import os

BAGEL_T2I_ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "4:3": (768, 1024),
    "3:4": (1024, 768),
    "16:9": (576, 1024),
    "9:16": (1024, 576),
}


def get_config_value(config, key, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def get_bagel_latent_downsample(config):
    explicit = get_config_value(config, "latent_downsample", None)
    if explicit is not None:
        return int(explicit)

    vae_config = get_config_value(config, "vae_config", None)
    if vae_config is None:
        raise ValueError("BAGEL config must include `vae_config.downsample` to resolve image shapes")

    vae_downsample = get_config_value(vae_config, "downsample", None)
    latent_patch_size = get_config_value(config, "latent_patch_size", None)
    if vae_downsample is None or latent_patch_size is None:
        raise ValueError("BAGEL config must include `vae_config.downsample` and `latent_patch_size` to resolve image shapes")
    return int(vae_downsample) * int(latent_patch_size)


def resolve_bagel_t2i_image_shape(input_info, config):
    latent_downsample = get_bagel_latent_downsample(config)
    target_shape = getattr(input_info, "target_shape", None) or []

    if len(target_shape) == 2:
        try:
            height, width = int(target_shape[0]), int(target_shape[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"BAGEL target_shape must be two positive integers [H W], got: {target_shape}") from exc
        if height <= 0 or width <= 0:
            raise ValueError(f"BAGEL target_shape must be positive [H W], got: {target_shape}")
        if height % latent_downsample != 0 or width % latent_downsample != 0:
            raise ValueError(f"BAGEL target_shape must be divisible by latent downsample {latent_downsample}, got: {[height, width]}")
        return (height, width)

    if target_shape:
        raise ValueError(f"BAGEL target_shape must be [H W] when set, got: {target_shape}")

    aspect_ratio = getattr(input_info, "aspect_ratio", None) or get_config_value(config, "aspect_ratio", None) or "1:1"
    if aspect_ratio not in BAGEL_T2I_ASPECT_RATIOS:
        raise ValueError(f"Unsupported BAGEL aspect_ratio: {aspect_ratio}. Available: {sorted(BAGEL_T2I_ASPECT_RATIOS)}")
    return BAGEL_T2I_ASPECT_RATIOS[aspect_ratio]


def validate_bagel_model_assets(config, model_path):
    required_keys = [
        "llm_config",
        "llm_config_update",
        "inference_hyper",
        "vae_config",
        "latent_patch_size",
        "max_latent_size_update",
        "visual_gen",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"BAGEL config missing required key(s): {missing}. Check model_path/config.json and configs/bagel/bagel_t2i.json.")

    required_files = ["ema.safetensors", "ae.safetensors"]
    missing_files = [name for name in required_files if not os.path.exists(os.path.join(model_path, name))]
    if missing_files:
        raise FileNotFoundError(f"BAGEL model_path is missing required file(s): {missing_files}. Expected ByteDance-Seed/BAGEL-7B-MoT layout under {model_path}.")
