from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lightx2v.utils.set_config import (
    MODEL_BUNDLE_PROTECTED_KEYS,
    get_default_config,
    load_model_config,
    update_from_model_bundle_config,
)


def _write_bundle_config(model_path: Path, relative_path: str, payload: dict[str, Any]) -> None:
    target = model_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


def test_protected_keys_cover_runner_selection_and_remote_code():
    assert {"model_cls", "trust_remote_code"} <= MODEL_BUNDLE_PROTECTED_KEYS


def test_update_from_model_bundle_config_drops_caller_controlled_keys():
    config = {"model_cls": "ltx2", "num_layers": 1}
    update_from_model_bundle_config(
        config,
        {"model_cls": "cosmos3", "trust_remote_code": True, "num_layers": 4},
    )
    assert config["model_cls"] == "ltx2"
    assert "trust_remote_code" not in config
    # Ordinary architecture fields still merge, so the fix stays non-breaking.
    assert config["num_layers"] == 4


def test_cosmos3_bundle_cannot_enable_trust_remote_code(tmp_path):
    """A bundle's transformer/config.json must not flip the trust decision."""
    _write_bundle_config(tmp_path, "transformer/config.json", {"trust_remote_code": True, "base_fps": 30})
    config = get_default_config()
    config["model_cls"] = "cosmos3"
    config["model_path"] = str(tmp_path)
    config["task"] = "t2v"

    load_model_config(config)

    assert "trust_remote_code" not in config
    assert config["base_fps"] == 30


def test_bundle_config_cannot_reselect_the_runner(tmp_path):
    """A model-side config.json must not replace the caller-selected runner."""
    _write_bundle_config(
        tmp_path,
        "config.json",
        {"model_cls": "cosmos3", "trust_remote_code": True, "num_layers": 4},
    )
    config = get_default_config()
    config["model_cls"] = "ltx2"
    config["model_path"] = str(tmp_path)
    config["task"] = "t2v"

    load_model_config(config)

    assert config["model_cls"] == "ltx2"
    assert "trust_remote_code" not in config
    assert config["num_layers"] == 4
