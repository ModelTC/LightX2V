"""Assemble startup configuration without loading the inference runtime."""

import json
import os

from loguru import logger

from lightx2v.models.runners.registry import RUNNER_MODULES, get_model_config_loader
from lightx2v.utils.image_io import validate_save_image_method
from lightx2v.utils.lockable_dict import LockableDict
from lightx2v.utils.request_defaults import get_request_default_arguments, validate_request_defaults


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
    context = model_config | deployment_config
    arguments = get_request_default_arguments(context) if context.get("model_cls") in RUNNER_MODULES else {}
    defaults = validate_request_defaults(deployment_config, arguments=arguments)
    misplaced = deployment_config.keys() & arguments.keys()
    if misplaced:
        raise ValueError(f"Configure request fields under request_defaults: {', '.join(sorted(misplaced))}")
    model_config = dict(model_config)
    model_defaults = dict(model_config.pop("request_defaults", {}))
    for name in arguments:
        if name in model_config:
            model_defaults[name] = model_config.pop(name)
    config = LockableDict(model_config | deployment_config | {"request_defaults": model_defaults | defaults})
    validate_save_image_method(config.get("save_image_method", "opencv"))
    if "infer_steps" not in config and "num_inference_steps" in config:
        config["infer_steps"] = config["num_inference_steps"]
    if os.getenv("PLATFORM", "cuda") == "mps":
        assert not (config.get("lora_mode", "merge") in ("separate", "consumer") or any(config.get(key) for key in ("lora_configs", "lora_path"))), (
            "MPS inference does not support LoRA configs, dynamic LoRA, or load-time LoRA merging. "
            "Merge LoRA into the model weights offline before inference, remove the LoRA settings, and load the merged model checkpoint."
        )
    return config
