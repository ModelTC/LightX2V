import json
import os

from lightx2v.models.runners.config import CHECKPOINT_SUBFOLDERS, load_quantized_config, load_vae_config, normalize_num_frames, read_config
from lightx2v.utils.model_paths import find_torch_model_path
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    model_path = deployment_config["model_path"]
    model_config = {}
    if deployment_config["model_cls"] == "ltx2_5":
        component_files = {
            "dit_original_ckpt": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
            "text_encoder_original_ckpt": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
            "video_vae_original_ckpt": "vae/ltx-2.5-video-vae-bf16.safetensors",
            "audio_vae_original_ckpt": "vae/ltx-2.5-audio-vae-bf16.safetensors",
            "upsampler_original_ckpt": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        }
        if deployment_config.get("auto_duration", True):
            component_files["duration_head_original_ckpt"] = "model_patches/ltx-2.5-duration-head-bf16.safetensors"
        for key, filename in component_files.items():
            model_config[key] = find_torch_model_path(deployment_config, key, filename, subdir=[])

        # The release root has no config.json; the authoritative transformer
        # architecture is embedded in the safetensors metadata.  Merge it in
        # just like the directory-based LTX-2 loader does, while retaining
        # LightX2V's rope implementation selector.
        transformer_path = model_config["dit_original_ckpt"]
        if transformer_path and os.path.isfile(transformer_path):
            try:
                from safetensors import safe_open

                with safe_open(transformer_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}
                checkpoint_config = json.loads(metadata.get("config", "{}"))
                transformer_config = dict(checkpoint_config.get("transformer", {}))
                transformer_config.pop("rope_type", None)
                model_config.update(transformer_config)
                model_config["ltx_model_version"] = metadata.get("model_version", "")
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError(f"Failed to read LTX-2.5 transformer metadata from {transformer_path}: {exc}") from exc

    for subfolder in CHECKPOINT_SUBFOLDERS:
        path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(path):
            checkpoint_config = read_config(path)
            if subfolder in ("", "transformer"):
                # Checkpoint rope_type describes its layout, not the runtime RoPE implementation.
                checkpoint_config.pop("rope_type", None)
            model_config.update(checkpoint_config)
            break
    model_config.update(load_quantized_config(deployment_config))
    model_config.update(load_vae_config(deployment_config))
    config = merge_config(model_config, deployment_config)
    config["vae_stride"] = config["vae_scale_factors"]
    normalize_num_frames(config)
    return config
