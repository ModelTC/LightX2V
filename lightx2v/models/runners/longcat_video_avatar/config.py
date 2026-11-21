from os.path import abspath
from pathlib import Path

from lightx2v.models.runners.config import read_config
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    root = Path(abspath(deployment_config["model_path"]))
    shared = root.parent / "LongCat-Video"
    component_paths = {
        "dit_original_ckpt": root / "base_model",
        "t5_original_ckpt": shared / "text_encoder",
        "t5_tokenizer_path": shared / "tokenizer",
        "vae_path": shared / "vae" / "diffusion_pytorch_model.safetensors",
        "audio_encoder_path": root / "whisper-large-v3",
        "vocal_separator_path": root / "vocal_separator" / "Kim_Vocal_2.onnx",
    }
    deployment_config = dict(deployment_config)
    for key, default_path in component_paths.items():
        path = deployment_config.get(key)
        deployment_config[key] = str(default_path if path is None else path)

    dit_path = Path(deployment_config["dit_original_ckpt"])
    model_config = read_config(dit_path / "config.json")
    if model_config.get("_class_name") != "LongCatVideoAvatarTransformer3DModel" or model_config["audio_prenorm"] or (dit_path / "quantization_config.json").is_file():
        raise ValueError(f"Expected a LongCat-Video-Avatar 1.6 BF16 checkpoint at {dit_path}")
    vae_config = read_config(Path(deployment_config["vae_path"]).with_name("config.json"))

    model_config.update(
        vae_scale_factor=2 ** (len(vae_config["dim_mult"]) - 1),
        vae_latents_mean=vae_config["latents_mean"],
        vae_latents_std=vae_config["latents_std"],
        sample_shift=read_config(root / "scheduler" / "scheduler_config.json")["shift"],
        infer_steps=8,
        fps=25,
        mix_steps=4,
        mix_alpha=1.0,
        attn_type="flash_attn3",
        audio_vocal_separation=True,
        request_defaults={"num_frames": 93},
    )
    config = merge_config(model_config, deployment_config)
    _validate_config(config)
    return config


def _validate_config(config):
    if type(config["infer_steps"]) is not int or config["infer_steps"] != 8 or config["fps"] != 25:
        raise ValueError("LongCat-Video-Avatar 1.6 uses 8 distillation steps and 25 fps")
    mix_steps, mix_alpha = config["mix_steps"], config["mix_alpha"]
    if type(mix_steps) is not int or not 1 <= mix_steps <= config["infer_steps"]:
        raise ValueError("mix_steps must be an integer in [1, 8]")
    if type(mix_alpha) not in (int, float) or not 0 <= mix_alpha <= 1:
        raise ValueError("mix_alpha must be a number in [0, 1]")
    for key in ("enable_cfg", "dit_quantized", "t5_quantized", "cpu_offload", "lazy_load", "unload_modules", "lora_configs", "model_variant"):
        if config.get(key):
            raise ValueError(f"LongCat-Video-Avatar BF16 distillation does not support {key}")
    if config.get("feature_caching", "NoCaching") != "NoCaching":
        raise ValueError("LongCat-Video-Avatar requires feature_caching=NoCaching")
    parallel = config.get("parallel") or {}
    if parallel.get("seq_p_size", 1) not in (1, 2) or any(parallel.get(key, 1) != 1 for key in ("cfg_p_size", "tensor_p_size", "pp_size")):
        raise ValueError("LongCat-Video-Avatar supports one device or two-way sequence parallelism")
    if parallel.get("seq_p_size") == 2 and parallel.get("seq_p_attn_type", "ulysses") != "ulysses":
        raise ValueError("LongCat-Video-Avatar SP2 requires parallel.seq_p_attn_type=ulysses")
    if config["attn_type"] not in ("flash_attn2", "flash_attn3"):
        raise ValueError("LongCat-Video-Avatar attn_type must be flash_attn2 or flash_attn3")
