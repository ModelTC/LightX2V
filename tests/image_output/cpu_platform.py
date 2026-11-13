"""Register a CPU ``PLATFORM`` so the MiniMax-H3-Image tests run without an accelerator.

LightX2V binds ``AI_DEVICE`` once, when ``lightx2v`` is first imported, so this
module must be imported before anything from ``lightx2v``. ``torch.cpu`` lacks
a few of the device-module helpers the runners call; they get no-op stand-ins.
"""

import os

import torch

os.environ["PLATFORM"] = "cpu"

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER  # noqa: E402

if "cpu" not in PLATFORM_DEVICE_REGISTER:

    @PLATFORM_DEVICE_REGISTER("cpu")
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


for _name, _value in (
    ("manual_seed", torch.manual_seed),
    ("manual_seed_all", torch.manual_seed),
    ("empty_cache", lambda: None),
):
    if not hasattr(torch.cpu, _name):
        setattr(torch.cpu, _name, _value)
