import json
import os
from dataclasses import fields

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.tensor.device_mesh import init_device_mesh

from lightx2v.utils.input_info import align_num_frames
from lightx2v.utils.lockable_dict import LockableDict
from lightx2v.utils.utils import find_torch_model_path, is_main_process
from lightx2v_platform.base.global_var import AI_DEVICE


def get_default_config():
    return LockableDict(
        {
            "do_mm_calib": False,
            "cpu_offload": False,
            "max_area": False,
            "vae_stride": (4, 8, 8),
            "patch_size": (1, 2, 2),
            "feature_caching": "NoCaching",  # ["NoCaching", "TaylorSeer", "Tea"]
            "teacache_thresh": 0.26,
            "use_ret_steps": False,
            "use_bfloat16": True,
            "lora_configs": None,  # List of dicts with 'path' and 'strength' keys
            "parallel": False,
            "seq_parallel": False,
            "cfg_parallel": False,
            "enable_cfg": False,
            "warmup": False,
            "use_image_encoder": True,
        }
    )


def build_startup_config(config_data):
    """Assemble startup settings from deployment and model configs."""
    config = get_default_config()
    config.update(config_data)
    if config.get("config_json") is not None:
        logger.info(f"Loading some config from {config['config_json']}")
        with open(config["config_json"], "r") as f:
            config_json = json.load(f)
        config.update(config_json)
    config["task"] = config_data.get("task")
    if config_data.get("model_variant") is not None:
        config["model_variant"] = config_data["model_variant"]
    if config_data.get("fps") is not None:
        config["fps"] = config_data["fps"]

    load_model_config(config)
    return config


def load_model_config(config):
    """Load model settings and normalize the startup configuration."""
    if config.get("model_cls") == "ltx2_5":
        # Match Wan's checkpoint lookup contract: an explicit component path
        # wins; otherwise find the released filename below --model_path.
        component_files = {
            "dit_original_ckpt": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
            "text_encoder_original_ckpt": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
            "video_vae_original_ckpt": "vae/ltx-2.5-video-vae-bf16.safetensors",
            "audio_vae_original_ckpt": "vae/ltx-2.5-audio-vae-bf16.safetensors",
            "duration_head_original_ckpt": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
            "upsampler_original_ckpt": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        }
        for key, filename in component_files.items():
            config[key] = find_torch_model_path(config, key, filename, subdir=[])

        # The release root has no config.json; the authoritative transformer
        # architecture is embedded in the safetensors metadata.  Merge it in
        # just like the directory-based LTX-2 loader does, while retaining
        # LightX2V's rope implementation selector.
        transformer_path = config.get("dit_original_ckpt")
        if transformer_path and os.path.isfile(transformer_path):
            try:
                from safetensors import safe_open

                with safe_open(transformer_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}
                checkpoint_config = json.loads(metadata.get("config", "{}"))
                transformer_config = dict(checkpoint_config.get("transformer", {}))
                transformer_config.pop("rope_type", None)
                config.update(transformer_config)
                config["ltx_model_version"] = metadata.get("model_version", "")
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError(f"Failed to read LTX-2.5 transformer metadata from {transformer_path}: {exc}") from exc

    assert os.path.exists(config["model_path"]), f"Model path not found: {config['model_path']}"

    if config["model_cls"] in {"hunyuan_video_1.5", "worldplay_distill", "worldplay_ar", "worldplay_bi"}:
        config["transformer_model_path"] = os.path.join(config["model_path"], "transformer", config["transformer_model_name"])
        if os.path.exists(os.path.join(config["transformer_model_path"], "config.json")):
            with open(os.path.join(config["transformer_model_path"], "config.json"), "r") as f:
                model_config = json.load(f)
            config.update(model_config)
    elif config["model_cls"] in {"hunyuan3d", "hidream_o1_image"}:
        # These runners load their own model configuration.
        pass
    elif config["model_cls"] == "sensenova_vision":
        llm_config_path = os.path.join(config["model_path"], "llm_config.json")
        vit_config_path = os.path.join(config["model_path"], "vit_config.json")
        missing = [path for path in (llm_config_path, vit_config_path) if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"SenseNova-Vision model_path must contain llm_config.json and vit_config.json; missing: {missing}")
        with open(llm_config_path, "r") as f:
            config["llm_config"] = json.load(f)
        with open(vit_config_path, "r") as f:
            config["vit_config"] = json.load(f)

        # SenseNova's root config.json is project metadata, not a Bagel model
        # config. Assemble the shared Bagel structure explicitly instead.
        config["vae_config"] = {"z_channels": 16, "downsample": 8}
        config["visual_gen"] = True
        config["visual_und"] = True
        config["latent_patch_size"] = 2
        config["max_latent_size_update"] = 64
        config["vit_max_num_patch_per_side"] = 70
        config["connector_act"] = "gelu_pytorch_tanh"
        config["interpolate_pos"] = False
        config["enable_vision_context"] = True
        sensenova_source_path = os.getenv("SENSENOVA_SOURCE_PATH", "").strip()
        if sensenova_source_path:
            config["sensenova_source_path"] = sensenova_source_path
    elif config["model_cls"] == "wan2.2_s2v":
        config_path = os.path.join(config["model_path"], "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config = json.load(f)
            config.update(model_config)
        config.setdefault("text_dim", 4096)
        config.setdefault("patch_size", (1, 2, 2))
        config.setdefault("window_size", (-1, -1))
        config.setdefault("qk_norm", True)
        config.setdefault("cross_attn_norm", True)
    elif config["model_cls"] == "worldmirror":
        # WorldMirror weights live under {model_path}/{subfolder}/, with a config.json
        # alongside model.safetensors. The runner loads this config directly; here we
        # only expose the resolved transformer path for logging.
        subfolder = config.get("subfolder", "HY-WorldMirror-2.0")
        candidate = os.path.join(config["model_path"], subfolder)
        if os.path.isdir(candidate):
            config["transformer_model_path"] = candidate
        else:
            config["transformer_model_path"] = config["model_path"]
    elif config["model_cls"] == "dreamzero":
        config_path = os.path.join(config["model_path"], "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config = json.load(f)
            config.update(model_config)
            action_head_config = model_config.get("action_head_cfg", {}).get("config", {})
            diffusion_model_config = action_head_config.get("diffusion_model_cfg", {})
            config.update(action_head_config)
            config.update(diffusion_model_config)
            if "out_dim" in config:
                config["num_channels_latents"] = config["out_dim"]
    elif config["model_cls"] == "longcat_image":
        for subfolder in ("", "transformer"):
            config_path = os.path.join(config["model_path"], subfolder, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config.update(json.load(f))
    elif config["model_cls"] in {"cosmos3", "lingbot_video"}:
        transformer_config_path = os.path.join(config["model_path"], "transformer", "config.json")
        if os.path.exists(transformer_config_path):
            with open(transformer_config_path, "r") as f:
                model_config = json.load(f)
            config.update(model_config)
        config.setdefault("num_frames", 1)
        config.setdefault("fps", config.get("base_fps", 24) if config["model_cls"] == "cosmos3" else 24)
        if config["model_cls"] == "lingbot_video":
            config.setdefault("vae_scale_factor_spatial", 8)
            config.setdefault("vae_scale_factor_temporal", 4)
            config.setdefault("vae_scale_factor", 8)
    elif config["model_cls"] == "minimax_h3":
        model_variant = config.get("model_variant")
        if model_variant not in ("fl2av", "ref2av"):
            raise ValueError("MiniMax-H3 requires model_variant='fl2av' or 'ref2av'; set --model-variant when starting the model")
        transformer_subfolder = "transformer_ref" if model_variant == "ref2av" else "transformer"
        transformer_path = os.path.join(config["model_path"], transformer_subfolder)
        transformer_config_path = os.path.join(transformer_path, "config.json")
        if not os.path.isfile(transformer_config_path):
            raise FileNotFoundError(f"MiniMax-H3 transformer config not found: {transformer_config_path}")
        with open(transformer_config_path, "r") as f:
            model_config = json.load(f)
        config.update(model_config)
        config["model_variant"] = model_variant
        config.pop("task", None)
        config["dit_original_ckpt"] = transformer_path
        if config.get("dit_quantized_ckpt"):
            config["dit_quantized"] = True
            config["dit_quant_scheme"] = {
                "fp8": "fp8-q8f",
                "int8": "int8-q8f",
            }.get(config.get("dit_quant_scheme"), config.get("dit_quant_scheme", "Default"))
        config["enable_cfg"] = False
        config["fps"] = 24
        config.setdefault("video_flow_shift", 12.0)
        config.setdefault("audio_flow_shift", 3.0)
        config.setdefault("audio_sampling_rate", 32000)
    else:
        for subfolder in ("", "low_noise_model", "distill_models/low_noise_model", "original", "transformer"):
            config_path = os.path.join(config["model_path"], subfolder, "config.json")
            if not os.path.exists(config_path):
                continue
            with open(config_path, "r") as f:
                model_config = json.load(f)
            if config["model_cls"] in {"ltx2", "ltx2_ar", "ltx2_5"} and subfolder in ("", "transformer"):
                # LTX uses rope_type for the layout, while LightX2V uses it
                # to select a registered RoPE implementation.
                model_config.pop("rope_type", None)
            elif config["model_cls"] == "z_image" and subfolder == "transformer":
                # https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/transformer/config.json
                z_image_patch_size = model_config.pop("all_patch_size", [2])
                z_image_f_patch_size = model_config.pop("all_f_patch_size", [1])
                if not (len(z_image_patch_size) == 1 and len(z_image_f_patch_size) == 1):
                    raise ValueError(
                        f"Expected 'all_patch_size' and 'all_f_patch_size' in z_image config to be lists of length 1, "
                        f"but got lengths {len(z_image_patch_size)} and {len(z_image_f_patch_size)} respectively. "
                        f"If the official z-image configs have been updated, ensure the current lightx2v's z-image model "
                        f"implementation matches the new configs then update this check."
                    )

                model_config["patch_size"] = z_image_patch_size[0]
                model_config["f_patch_size"] = z_image_f_patch_size[0]

            config.update(model_config)
            break
        # load quantized config
        if config.get("dit_quantized_ckpt") is not None:
            config_path = os.path.join(config["dit_quantized_ckpt"], "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    model_config = json.load(f)
                config.update(model_config)

    vae_config_path = os.path.join(config["model_path"], "vae", "config.json")
    if os.path.exists(vae_config_path):
        with open(vae_config_path, "r") as f:
            vae_config = json.load(f)
        if "temperal_downsample" in vae_config:
            config["vae_scale_factor"] = 2 ** len(vae_config["temperal_downsample"])
        elif "block_out_channels" in vae_config:
            config["vae_scale_factor"] = 2 ** (len(vae_config["block_out_channels"]) - 1)
        if config["model_cls"] == "ernie_image":
            config["vae_scale_factor"] = 2 ** len(vae_config["block_out_channels"])
        elif config["model_cls"] == "cosmos3":
            config["vae_scale_factor_spatial"] = int(vae_config.get("scale_factor_spatial", 16))
            config["vae_scale_factor_temporal"] = int(vae_config.get("scale_factor_temporal", 4))
            config["vae_scale_factor"] = config["vae_scale_factor_spatial"]

    if config["model_cls"] == "lingbot_video":
        config["vae_scale_factor_spatial"] = int(config.get("vae_scale_factor_spatial", 8))
        config["vae_scale_factor_temporal"] = int(config.get("vae_scale_factor_temporal", 4))
        config["vae_scale_factor"] = config["vae_scale_factor_spatial"]
    if config["model_cls"] == "minimax_h3":
        # The generic Diffusers-VAE heuristic above counts six encoder stages
        # and would incorrectly derive 32. H3 downsamples space by exactly 16.
        config["vae_spatial_scale_factor"] = 16
        config["vae_scale_factor"] = 16
    if config["model_cls"] == "cosmos3" and os.path.exists(os.path.join(config["model_path"], "sound_tokenizer", "config.json")):
        with open(os.path.join(config["model_path"], "sound_tokenizer", "config.json"), "r") as f:
            sound_config = json.load(f)
        config["sound_sampling_rate"] = int(sound_config.get("sampling_rate", 48000))
        config["sound_hop_size"] = int(sound_config.get("hop_size", 1920))

    # Some upstream/offical configs use `num_inference_steps`, while the shared
    # LightX2V scheduler stack expects `infer_steps`.
    if "infer_steps" not in config and "num_inference_steps" in config:
        config["infer_steps"] = config["num_inference_steps"]

    if config["model_cls"] == "hunyuan_image3":
        from lightx2v.models.networks.hunyuan_image3.config import normalize_hunyuan_image3_config

        normalize_hunyuan_image3_config(config)

    if config["model_cls"] == "lingbot_va" and "num_frames" not in config:
        ar_config = config.get("ar_config", {})
        required_keys = ("num_frame_per_chunk", "num_chunks")
        missing_keys = [key for key in required_keys if key not in ar_config]
        if missing_keys:
            raise ValueError(f"LingBot-VA requires ar_config.{', ar_config.'.join(missing_keys)} to derive num_frames.")
        latent_frames = int(ar_config["num_frame_per_chunk"]) * int(ar_config["num_chunks"])
        temporal_stride = int(config["vae_stride"][0])
        if latent_frames <= 0 or temporal_stride <= 0:
            raise ValueError(f"LingBot-VA requires positive latent frame count and VAE temporal stride, got latent_frames={latent_frames}, temporal_stride={temporal_stride}.")
        config["num_frames"] = (latent_frames - 1) * temporal_stride + 1
        logger.info(f"Auto-set LingBot-VA num_frames={config['num_frames']} from {latent_frames} latent frames and temporal stride {temporal_stride}.")

    if config["model_cls"] != "minimax_h3" and config["task"] in ["i2v", "t2av", "i2av", "i2va", "s2v", "rs2v", "ltx2_s2v", "v2av"] and config.get("num_frames") is not None and "vae_stride" in config:
        temporal_stride = int(config["vae_stride"][0])
        if (config["num_frames"] - 1) % temporal_stride != 0:
            original_length = config["num_frames"]
            config["num_frames"] = align_num_frames(original_length, temporal_stride)
            logger.warning(f"`num_frames - 1` must be divisible by {temporal_stride}; using {config['num_frames']} instead of {original_length}.")


def build_cli_inputs(args):
    args_data = {key: value for key, value in vars(args).items() if value is not None}
    startup_fields = {"config_json", "model_cls", "model_variant", "model_path", "task"}
    startup_args = {key: value for key, value in args_data.items() if key in startup_fields}
    request_data = {key: value for key, value in args_data.items() if key not in startup_fields}
    request_data["task"] = args.task
    startup_config = build_startup_config(startup_args)
    return startup_config, request_data


def init_parallel(config):
    """Create the model's parallel mesh and warm up its communication."""
    parallel = config["parallel"]
    if not parallel:
        return

    tensor_p_size = int(parallel.get("tensor_p_size", 1))
    cfg_p_size = int(parallel.get("cfg_p_size", 1))
    seq_p_size = int(parallel.get("seq_p_size", 1))
    world_size = dist.get_world_size()
    expected_world_size = tensor_p_size * cfg_p_size * seq_p_size
    if expected_world_size != world_size:
        raise ValueError(f"Parallel sizes must match the distributed world size: tensor_p_size ({tensor_p_size}) * cfg_p_size ({cfg_p_size}) * seq_p_size ({seq_p_size}) != world_size ({world_size}).")

    if config.get("model_cls") == "hunyuan_image3" and parallel.get("phase_aware", False):
        from lightx2v.models.networks.hunyuan_image3.parallel import initialize_hunyuan_image3_parallel_runtime

        initialize_hunyuan_image3_parallel_runtime(config)
    else:
        # Keep the original CFG/SP dimensions without TP. With TP, omit unit
        # dimensions and place TP last so its ranks form contiguous groups.
        mesh_dims = [("cfg_p", cfg_p_size), ("seq_p", seq_p_size)]
        if tensor_p_size > 1:
            mesh_dims = [(name, size) for name, size in mesh_dims if size > 1]
            mesh_dims.append(("tensor_p", tensor_p_size))
        config["device_mesh"] = init_device_mesh(
            AI_DEVICE,
            tuple(size for _, size in mesh_dims),
            mesh_dim_names=tuple(name for name, _ in mesh_dims),
        )
        config["tensor_parallel"] = tensor_p_size > 1
        config["seq_parallel"] = seq_p_size > 1
        config["cfg_parallel"] = bool(config.get("enable_cfg", False) and cfg_p_size > 1)

    warmup_device = f"cuda:{torch.cuda.current_device()}" if AI_DEVICE == "cuda" else AI_DEVICE
    warmup_tensor = torch.zeros([1], device=warmup_device)
    dist.all_reduce(warmup_tensor)


def print_config(config, title="config"):
    if is_main_process():
        logger.info(f"{title}:\n{json.dumps(config, ensure_ascii=False, indent=4, default=str)}")


def print_request(input_info, supported_request_fields):
    request = {input_field.name: getattr(input_info, input_field.name) for input_field in fields(input_info) if input_field.repr and input_field.name in supported_request_fields}
    print_config(request, title="Effective request")
