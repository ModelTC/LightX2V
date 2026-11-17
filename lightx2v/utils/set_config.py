"""Assemble startup configuration without loading the inference runtime."""

import json
import os

from loguru import logger

from lightx2v.models.runners.registry import get_model_config_loader
from lightx2v.utils.image_io import validate_save_image_method
from lightx2v.utils.lockable_dict import LockableDict


def build_startup_config(config_data, *, configure_runtime=True):
    deployment_config = dict(config_data)
    if deployment_config.get("config_json") is not None:
        logger.info(f"Loading some config from {deployment_config['config_json']}")
        with open(deployment_config["config_json"], "r") as f:
            deployment_config.update(json.load(f))
    deployment_config["task"] = config_data.get("task")
    for name in ("model_variant", "fps"):
        if config_data.get(name) is not None:
            deployment_config[name] = config_data[name]

    if not os.path.exists(deployment_config["model_path"]):
        raise FileNotFoundError(f"Model path not found: {deployment_config['model_path']}")
    load_config = get_model_config_loader(deployment_config["model_cls"])
    config = load_config(deployment_config)
    if configure_runtime:
        from lightx2v.utils.parallel import configure_nccl_graph

        configure_nccl_graph(config)
    return config


def merge_config(model_config, deployment_config):
    config = LockableDict(model_config | deployment_config)
    validate_save_image_method(config.get("save_image_method", "opencv"))
    if "infer_steps" not in config and "num_inference_steps" in config:
        config["infer_steps"] = config["num_inference_steps"]
    if os.getenv("PLATFORM", "cuda") == "mps":
        assert not (config.get("lora_mode", "merge") in ("separate", "consumer") or any(config.get(key) for key in ("lora_configs", "lora_path"))), (
            "MPS inference does not support LoRA configs, dynamic LoRA, or load-time LoRA merging. "
            "Merge LoRA into the model weights offline before inference, remove the LoRA settings, and load the merged model checkpoint."
        )
    return config
