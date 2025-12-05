import os

from lightx2v_platform import *


def set_ai_device():
    platform = os.getenv("PLATFORM", "cuda")
    init_ai_device(platform)
    from lightx2v_platform.base.global_var import AI_DEVICE

    check_ai_device(AI_DEVICE)


set_ai_device()
from lightx2v_platform.ops import *  # noqa: E402
