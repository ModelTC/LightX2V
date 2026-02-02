import json
import logging
import os
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional

from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.utils import find_torch_model_path

from lightx2v.models.input_encoders.hf.wan.t5.model import T5EncoderModel
from lightx2v.models.input_encoders.hf.wan.xlm_roberta.model import CLIPModel
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v.models.video_encoders.hf.wan.vae_tiny import WanVAE_tiny, Wan2_2_VAE_tiny
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.utils.set_config import get_default_config, set_config as set_config_base

logger = logging.getLogger(__name__)

class ConfigObj:
    """Helper class to convert dictionary to object with attributes"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def set_config(
        model_path,
        task,
        model_cls,
        config_path=None,
        **kwargs
    ):
    """
    Load configuration for Wan model.
    """
    # Create arguments object similar to what set_config expects
    args_dict = {
        "task": task,
        "model_path": model_path,
        "model_cls": model_cls,
        "config_json": config_path,
    }
    args_dict.update(kwargs)
    
    # Convert to object for set_config compatibility
    args = ConfigObj(**args_dict)
    
    # Use existing set_config from utils
    config = set_config_base(args)
    
    return config

def build_wan_model_with_lora(wan_module, config, model_kwargs, lora_configs, model_type="high_noise_model"):
    lora_dynamic_apply = config.get("lora_dynamic_apply", False)

    if lora_dynamic_apply:
        if model_type in ["high_noise_model", "low_noise_model"]:
            # For wan2.2
            lora_name_to_info = {item["name"]: item for item in lora_configs}
            lora_path = lora_name_to_info[model_type]["path"]
            lora_strength = lora_name_to_info[model_type]["strength"]
        else:
            # For wan2.1
            lora_path = lora_configs[0]["path"]
            lora_strength = lora_configs[0]["strength"]

        model_kwargs["lora_path"] = lora_path
        model_kwargs["lora_strength"] = lora_strength
        model = wan_module(**model_kwargs)
    else:
        assert not config.get("dit_quantized", False), "Online LoRA only for quantized models; merging LoRA is unsupported."
        assert not config.get("lazy_load", False), "Lazy load mode does not support LoRA merging."
        model = wan_module(**model_kwargs)
        lora_wrapper = WanLoraWrapper(model)
        if model_type in ["high_noise_model", "low_noise_model"]:
            lora_configs = [lora_config for lora_config in lora_configs if lora_config["name"] == model_type]
        lora_wrapper.apply_lora(lora_configs, model_type=model_type)
    return model

def load_wan_text_encoder(config: Dict[str, Any]):
    # offload config
    t5_offload = config.get("t5_cpu_offload", config.get("cpu_offload"))
    if t5_offload:
        t5_device = torch.device("cpu")
    else:
        t5_device = torch.device(AI_DEVICE)
    tokenizer_path = os.path.join(config["model_path"], "google/umt5-xxl")
    # quant_config
    t5_quantized = config.get("t5_quantized", False)
    if t5_quantized:
        t5_quant_scheme = config.get("t5_quant_scheme", None)
        assert t5_quant_scheme is not None
        tmp_t5_quant_scheme = t5_quant_scheme.split("-")[0]
        t5_model_name = f"models_t5_umt5-xxl-enc-{tmp_t5_quant_scheme}.pth"
        t5_quantized_ckpt = find_torch_model_path(config, "t5_quantized_ckpt", t5_model_name)
        t5_original_ckpt = None
    else:
        t5_quant_scheme = None
        t5_quantized_ckpt = None
        t5_model_name = "models_t5_umt5-xxl-enc-bf16.pth"
        t5_original_ckpt = find_torch_model_path(config, "t5_original_ckpt", t5_model_name)

    text_encoder = T5EncoderModel(
        text_len=config["text_len"],
        dtype=torch.bfloat16,
        device=t5_device,
        checkpoint_path=t5_original_ckpt,
        tokenizer_path=tokenizer_path,
        shard_fn=None,
        cpu_offload=t5_offload,
        t5_quantized=t5_quantized,
        t5_quantized_ckpt=t5_quantized_ckpt,
        quant_scheme=t5_quant_scheme,
        load_from_rank0=config.get("load_from_rank0", False),
        lazy_load=config.get("t5_lazy_load", False),
    )
    # Return single encoder to match original returning list
    text_encoders = [text_encoder]
    return text_encoders

def load_wan_image_encoder(config: Dict[str, Any]):
    image_encoder = None
    if config["task"] in ["i2v", "flf2v", "animate", "s2v"] and config.get("use_image_encoder", True):
        # offload config
        clip_offload = config.get("clip_cpu_offload", config.get("cpu_offload", False))
        if clip_offload:
            clip_device = torch.device("cpu")
        else:
            clip_device = torch.device(AI_DEVICE)
        # quant_config
        clip_quantized = config.get("clip_quantized", False)
        if clip_quantized:
            clip_quant_scheme = config.get("clip_quant_scheme", None)
            assert clip_quant_scheme is not None
            tmp_clip_quant_scheme = clip_quant_scheme.split("-")[0]
            clip_model_name = f"models_clip_open-clip-xlm-roberta-large-vit-huge-14-{tmp_clip_quant_scheme}.pth"
            clip_quantized_ckpt = find_torch_model_path(config, "clip_quantized_ckpt", clip_model_name)
            clip_original_ckpt = None
        else:
            clip_quantized_ckpt = None
            clip_quant_scheme = None
            clip_model_name = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
            clip_original_ckpt = find_torch_model_path(config, "clip_original_ckpt", clip_model_name)

        image_encoder = CLIPModel(
            dtype=torch.float16,
            device=clip_device,
            checkpoint_path=clip_original_ckpt,
            clip_quantized=clip_quantized,
            clip_quantized_ckpt=clip_quantized_ckpt,
            quant_scheme=clip_quant_scheme,
            cpu_offload=clip_offload,
            use_31_block=config.get("use_31_block", True),
            load_from_rank0=config.get("load_from_rank0", False),
        )

    return image_encoder

def get_vae_parallel(config: Dict[str, Any]):
    if isinstance(config.get("parallel", False), bool):
        return config.get("parallel", False)
    if isinstance(config.get("parallel", False), dict):
        return config.get("parallel", {}).get("vae_parallel", True)
    return False

def load_wan_vae_encoder(config: Dict[str, Any]):
    vae_name = config.get("vae_name", "Wan2.1_VAE.pth")
    if config.get("model_cls", "") == "wan2.2":
        vae_cls = Wan2_2_VAE
    else:
        vae_cls = WanVAE

    # offload config
    vae_offload = config.get("vae_cpu_offload", config.get("cpu_offload"))
    if vae_offload:
        vae_device = torch.device("cpu")
    else:
        vae_device = torch.device(AI_DEVICE)

    vae_config = {
        "vae_path": find_torch_model_path(config, "vae_path", vae_name),
        "device": vae_device,
        "parallel": get_vae_parallel(config),
        "use_tiling": config.get("use_tiling_vae", False),
        "cpu_offload": vae_offload,
        "dtype": GET_DTYPE(),
        "load_from_rank0": config.get("load_from_rank0", False),
        "use_lightvae": config.get("use_lightvae", False),
    }
    if config["task"] not in ["i2v", "flf2v", "animate", "vace", "s2v"]:
        return None
    else:
        return vae_cls(**vae_config)

def load_wan_vae_decoder(config: Dict[str, Any]):
    
    vae_name = config.get("vae_name", "Wan2.1_VAE.pth")
    tiny_vae_name = "taew2_1.pth"
    
    if config.get("model_cls", "") == "wan2.2":
        vae_cls = Wan2_2_VAE
        tiny_vae_cls = Wan2_2_VAE_tiny
        tiny_vae_name = "taew2_2.pth"
    else:
        vae_cls = WanVAE
        tiny_vae_cls = WanVAE_tiny
        tiny_vae_name = "taew2_1.pth"

    # offload config
    vae_offload = config.get("vae_cpu_offload", config.get("cpu_offload"))
    if vae_offload:
        vae_device = torch.device("cpu")
    else:
        vae_device = torch.device(AI_DEVICE)

    vae_config = {
        "vae_path": find_torch_model_path(config, "vae_path", vae_name),
        "device": vae_device,
        "parallel": get_vae_parallel(config),
        "use_tiling": config.get("use_tiling_vae", False),
        "cpu_offload": vae_offload,
        "use_lightvae": config.get("use_lightvae", False),
        "dtype": GET_DTYPE(),
        "load_from_rank0": config.get("load_from_rank0", False),
    }
    if config.get("use_tae", False):
        tae_path = find_torch_model_path(config, "tae_path", tiny_vae_name)
        vae_decoder = tiny_vae_cls(vae_path=tae_path, device=AI_DEVICE, need_scaled=config.get("need_scaled", False)).to(AI_DEVICE)
    else:
        vae_decoder = vae_cls(**vae_config)
    return vae_decoder

def load_wan_transformer(config: Dict[str, Any]):
    if config["cpu_offload"]:
        init_device = torch.device("cpu")
    else:
        init_device = torch.device(AI_DEVICE)
    wan_model_kwargs = {"model_path": config["model_path"], "config": config, "device": init_device}
    lora_configs = config.get("lora_configs")
    if not lora_configs:
        model = WanModel(**wan_model_kwargs)
    else:
        model = build_wan_model_with_lora(WanModel, config, wan_model_kwargs, lora_configs, model_type="wan2.1")
    return model
