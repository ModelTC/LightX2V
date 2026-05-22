from pathlib import Path

import yaml
from omegaconf import OmegaConf


def load_config(path: str):
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = OmegaConf.create(raw)
    result = OmegaConf.to_container(config, resolve=True)
    result["config_path"] = str(resolved)
    return result
