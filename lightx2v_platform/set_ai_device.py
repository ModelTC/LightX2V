import os

from lightx2v_platform import *


def set_ai_device():
    platform = os.getenv("PLATFORM", "cuda")
    init_ai_device(platform)
    check_ai_device(platform)


set_ai_device()
from lightx2v_platform.ops import *  # noqa: E402
