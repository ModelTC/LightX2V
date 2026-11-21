import os

from lightx2v.models.runners.config import load_vae_config, read_config
from lightx2v.models.runners.wan.config import WAN_DEFAULTS
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    model_path = deployment_config["model_path"]
    model_cls = deployment_config["model_cls"]
    model_config = dict(WAN_DEFAULTS)
    is_wan21 = model_cls.startswith("wan21_")
    model_size = "14b" if is_wan21 else "5b"
    vae_filename = "Wan2.1_VAE.pth" if is_wan21 else "Wan2.2_VAE.pth"
    dit_subfolder = f"dit-{model_size}"
    if not model_cls.endswith("_ar"):
        dit_subfolder += "-bidir"
    component_files = {
        "dit_original_ckpt": dit_subfolder,
        "vae_path": f"vae-{model_size}/{vae_filename}",
        "t5_original_ckpt": "umt5/text_encoder/model.safetensors",
        "t5_tokenizer_path": "umt5/tokenizer",
    }
    if is_wan21:
        component_files["clip_original_ckpt"] = "clip/image_encoder/model.safetensors"
    for key, filename in component_files.items():
        model_config[key] = deployment_config.get(key, os.path.join(model_path, filename))
    dit_path = model_config["dit_original_ckpt"]
    dit_dir = dit_path if os.path.isdir(dit_path) else os.path.dirname(dit_path)
    checkpoint_config = read_config(os.path.join(dit_dir, "config.json"))
    model_config.update(checkpoint_config)
    model_config.setdefault("text_len", 512)
    model_config.setdefault("rope_type", "flashinfer_rope")
    model_config.setdefault("modulate_type", "triton")
    model_config.update(
        dim=checkpoint_config["num_attention_heads"] * checkpoint_config["attention_head_dim"],
        num_heads=checkpoint_config["num_attention_heads"],
        # Released configs record the base input width, excluding the trained proxy channels.
        in_dim=checkpoint_config["in_channels"] + 1024,
        out_dim=checkpoint_config["out_channels"],
        num_channels_latents=checkpoint_config["out_channels"],
    )
    vae_config_path = os.path.join(os.path.dirname(model_config["vae_path"]), "config.json")
    vae_config = read_config(vae_config_path)
    temporal_scale = vae_config.get("scale_factor_temporal", 2 ** sum(vae_config["temperal_downsample"]))
    spatial_scale = vae_config.get("scale_factor_spatial", 2 ** (len(vae_config["dim_mult"]) - 1) * vae_config.get("patch_size", 1))
    model_config["vae_stride"] = [temporal_scale, spatial_scale, spatial_scale]

    model_config.update(load_vae_config(deployment_config))
    return merge_config(model_config, deployment_config)
