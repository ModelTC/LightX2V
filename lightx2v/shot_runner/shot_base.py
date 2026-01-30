import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config


def load_clip_configs(main_json_path: str):
    with open(main_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lightx2v_path = cfg["lightx2v_path"]
    clip_configs_raw = cfg["clip_configs"]

    clip_configs = [ClipConfig(name=item["name"], config_json=str(Path(lightx2v_path) / item["path"])) for item in clip_configs_raw]

    return clip_configs


@dataclass
class ClipConfig:
    name: str
    config_json: str | dict[str, Any]


@dataclass
class ShotConfig:
    seed: int
    image_path: str
    audio_path: str
    prompt: str
    negative_prompt: str
    save_result_path: str
    clip_configs: list[ClipConfig]
    target_shape: list[int]


class ShotPipeline:
    def __init__(self, shot_cfg: ShotConfig):
        self.shot_cfg = shot_cfg
        self.clip_generators = {}
        self.clip_inputs = {}
        self.overlap_frame = None
        self.overlap_latent = None

        for clip_config in shot_cfg.clip_configs:
            name = clip_config.name
            self.clip_generators[name] = self.create_clip_generator(clip_config)

            args = Namespace(
                seed=self.shot_cfg.seed,
                prompt=self.shot_cfg.prompt,
                negative_prompt=self.shot_cfg.negative_prompt,
                image_path=self.shot_cfg.image_path,
                audio_path=self.shot_cfg.audio_path,
                save_result_path=self.shot_cfg.save_result_path,
                task=self.clip_generators[name].task,
                return_result_tensor=True,
                overlap_frame=self.overlap_frame,
                overlap_latent=self.overlap_latent,
                target_shape=self.shot_cfg.target_shape,
            )
            input_info = init_empty_input_info(self.clip_generators[name].task)
            update_input_info_from_dict(input_info, vars(args))
            self.clip_inputs[name] = input_info

    def _init_runner(self, config):
        torch.set_grad_enabled(False)
        runner = RUNNER_REGISTER[config["model_cls"]](config)
        runner.init_modules()
        return runner

    def create_clip_generator(self, clip_config: ClipConfig):
        config = self.get_config_json(clip_config.config_json)
        config = set_config(Namespace(**config))
        print_config(config)

        runner = self._init_runner(config)
        logger.info(f"Clip {clip_config.name} initialized successfully!")
        return runner

    def get_config_json(self, config_json):
        if isinstance(config_json, dict):
            logger.info("Using infer config from dict")
            return config_json
        if isinstance(config_json, str):
            logger.info(f"Loading infer config from {config_json}")
            with open(config_json, "r") as f:
                config = json.load(f)
            return config
        raise TypeError("config_json must be str or dict")

    @torch.no_grad()
    def generate(self):
        pass
