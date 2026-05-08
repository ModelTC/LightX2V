from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf


def load_config(path: str):
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = OmegaConf.create(raw)
    return OmegaConf.to_container(config, resolve=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
