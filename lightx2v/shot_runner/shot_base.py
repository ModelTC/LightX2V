import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from lightx2v.utils.input_info import fill_input_info_from_defaults
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
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


def load_clip_configs(main_json_path: str):
    with open(main_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "parallel" in cfg:
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None)
        platform_device.init_parallel_env()

    lightx2v_path = cfg["lightx2v_path"]
    clip_configs_raw = cfg["clip_configs"]

    clip_configs = []
    for item in clip_configs_raw:
        if "config" in item:
            config = item["config"]
        else:
            config_json = str(Path(lightx2v_path) / item["path"])
            config_json = {"config_json": config_json}
            config = set_config(Namespace(**config_json))

        if "parallel" in cfg:  # Add parallel config to clip json
            config["parallel"] = cfg["parallel"]
            set_parallel_config(config)

        clip_configs.append(ClipConfig(name=item["name"], config_json=config))
    return clip_configs


class ShotPipeline:
    def __init__(self, clip_configs: list[ClipConfig]):
        # self.clip_configs = clip_configs
        self.clip_generators = {}
        self.clip_inputs = {}
        self.progress_callback = None

        for clip_config in clip_configs:
            name = clip_config.name
            self.clip_generators[name] = self.create_clip_generator(clip_config)

    def check_input_info(self, user_input_info, clip_config):
        default_input_info = clip_config.get("default_input_info", None)
        if default_input_info is not None:
            fill_input_info_from_defaults(user_input_info, default_input_info)
        return user_input_info.normalize_unset_to_none()

    def _input_data_to_dict(self, input_data):
        if isinstance(input_data, dict):
            return input_data
        if hasattr(input_data, "__dict__"):
            return vars(input_data)
        return {}

    def update_input_info(self, input_data):
        data = self._input_data_to_dict(input_data)
        if not data:
            return

        # 将外部输入同步到 shot_cfg 和各 clip 的 input_info
        for key in ["seed", "image_path", "audio_path", "prompt", "negative_prompt", "save_result_path", "target_shape"]:
            if key in data and data[key] is not None:
                setattr(self.shot_cfg, key, data[key])

        for clip_input in self.clip_inputs.values():
            update_input_info_from_dict(clip_input, data)
            if hasattr(clip_input, "overlap_frame"):
                clip_input.overlap_frame = None
            if hasattr(clip_input, "overlap_latent"):
                clip_input.overlap_latent = None
            if hasattr(clip_input, "audio_clip"):
                clip_input.audio_clip = None

    def _init_runner(self, config):
        torch.set_grad_enabled(False)
        runner = RUNNER_REGISTER[config["model_cls"]](config)
        runner.init_modules()
        return runner

    def set_config(self, config_modify):
        for runner in self.clip_generators.values():
            if hasattr(runner, "set_config"):
                runner.set_config(config_modify)

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def create_clip_generator(self, clip_config: ClipConfig):
        logger.info(f"Clip {clip_config.name} initializing ... ")
        print_config(clip_config.config_json)
        runner = self._init_runner(clip_config.config_json)
        logger.info(f"Clip {clip_config.name} initialized successfully!")

        return runner

    @torch.no_grad()
    def generate(self):
        pass

    def run_pipeline(self, input_info):
        self.update_input_info(input_info)
        return self.generate()
