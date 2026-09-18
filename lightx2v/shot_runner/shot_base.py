import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from lightx2v.models.runners.runner_factory import build_runner
from lightx2v.utils.input_info import InputInfo, SekoTalkInputs
from lightx2v.utils.profiler import *
from lightx2v.utils.set_config import build_startup_config, init_parallel, print_config
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


@dataclass
class ClipConfig:
    name: str
    config_json: dict[str, Any]


def get_config_json(config_json):
    if isinstance(config_json, dict):
        logger.info("Using infer config from dict")
        return config_json
    if isinstance(config_json, str):
        logger.info(f"Loading infer config from {config_json}")
        with open(config_json, "r") as f:
            config = json.load(f)
        return config
    raise TypeError("config_json must be str or dict")


def load_clip_configs(main_json_path):
    if isinstance(main_json_path, dict):
        cfg = main_json_path
    else:
        with open(main_json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    if "parallel" in cfg:
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        platform_device.init_parallel_env()

    lightx2v_path = cfg["lightx2v_path"]
    clip_configs_raw = cfg["clip_configs"]

    clip_configs = []
    for item in clip_configs_raw:
        config_json = item["config"] if "config" in item else str(Path(lightx2v_path) / item["path"])
        config = build_startup_config(get_config_json(config_json))

        if "parallel" in cfg:  # Add parallel config to clip json
            config["parallel"] = cfg["parallel"]
            init_parallel(config)

        clip_configs.append(ClipConfig(name=item["name"], config_json=config))
    return clip_configs


class ShotPipeline:
    def __init__(self, clip_configs: list[ClipConfig]):
        self.clip_generators = {}
        self.progress_callback = None
        self.clip_name = clip_configs[0].name
        self.va_controller = None

        for clip_config in clip_configs:
            name = clip_config.name
            self.clip_generators[name] = self.create_clip_generator(clip_config)

    def prepare_request_data(self, input_data, runner):
        """Extract request fields from Shot inputs."""
        request_data = input_data if isinstance(input_data, dict) else vars(input_data)
        if isinstance(input_data, InputInfo):
            supported_request_fields = runner.get_supported_request_fields(runner.config["task"])
            # Legacy contexts use "" for omitted negative prompts.
            if isinstance(input_data, SekoTalkInputs) or request_data.get("negative_prompt"):
                supported_request_fields = supported_request_fields | {"negative_prompt"}
            request_data = {key: value for key, value in request_data.items() if key in supported_request_fields}
        return request_data

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def create_clip_generator(self, clip_config: ClipConfig):
        logger.info(f"Clip {clip_config.name} initializing ... ")
        print_config(clip_config.config_json)
        runner = build_runner(clip_config.config_json)
        logger.info(f"Clip {clip_config.name} initialized successfully!")

        return runner

    @torch.no_grad()
    def generate(self, args):
        raise NotImplementedError

    def run_pipeline(self, input_info):
        return self.generate(input_info)

    @property
    def config(self):
        return self.clip_generators[self.clip_name].config

    @property
    def stop_signal(self):
        return self.clip_generators[self.clip_name].stop_signal

    @stop_signal.setter
    def stop_signal(self, value):
        self.clip_generators[self.clip_name].stop_signal = value
