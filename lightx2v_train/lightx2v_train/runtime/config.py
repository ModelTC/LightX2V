from pathlib import Path

import yaml
from omegaconf import OmegaConf


def validate_block_indices(indices, block_count, option):
    if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < block_count for i in indices):
        raise ValueError(f"{option} must be a list of zero-based block indices below {block_count}.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{option} contains duplicate block indices.")
    return set(indices)


def load_config(path: str):
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = OmegaConf.create(raw)
    result = OmegaConf.to_container(config, resolve=True)
    result["config_path"] = str(resolved)
    return result
