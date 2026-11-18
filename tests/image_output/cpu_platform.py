"""Register a test-only CPU backend through platform plugin discovery."""

import importlib.metadata
import os
from unittest.mock import patch

import torch

os.environ["PLATFORM"] = "cpu"


class CpuDevice:
    name = "cpu"

    @staticmethod
    def init_device_env():
        pass

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device():
        return "cpu"


class CpuPlugin:
    name = "cpu"

    @staticmethod
    def load():
        def register():
            from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

            PLATFORM_DEVICE_REGISTER("cpu")(CpuDevice)

        return register


original_entry_points = importlib.metadata.entry_points


def entry_points(*args, **kwargs):
    if kwargs.get("group") == "lightx2v.platform_plugins":
        return [CpuPlugin()]
    return original_entry_points(*args, **kwargs)


with patch("importlib.metadata.entry_points", entry_points):
    import lightx2v_platform  # noqa: F401

for _name, _value in (
    ("manual_seed", torch.manual_seed),
    ("manual_seed_all", torch.manual_seed),
    ("empty_cache", lambda: None),
):
    if not hasattr(torch.cpu, _name):
        setattr(torch.cpu, _name, _value)
