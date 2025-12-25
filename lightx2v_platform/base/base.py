import os

from loguru import logger

from lightx2v_platform.base import global_var
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def init_ai_device(platform="cuda"):
    platform_device = PLATFORM_DEVICE_REGISTER.get(platform, None)
    if platform_device is None:
        available_platforms = list(PLATFORM_DEVICE_REGISTER.keys())
        raise RuntimeError(f"Unsupported PLATFORM: {platform}. Available PLATFORM: {available_platforms}")
    global_var.AI_DEVICE = platform_device.get_device()
    platform_device.init_device_env()
    logger.info(f"Initialized AI_DEVICE: {global_var.AI_DEVICE}")
    return global_var.AI_DEVICE


def check_ai_device(platform="cuda"):
    platform_device = PLATFORM_DEVICE_REGISTER.get(platform, None)
    if platform_device is None:
        available_platforms = list(PLATFORM_DEVICE_REGISTER.keys())
        raise RuntimeError(f"Unsupported platform: {platform}. Available platforms: {available_platforms}")
    is_available = platform_device.is_available()
    if not is_available:
        skip_platform_check = os.getenv("SKIP_PLATFORM_CHECK", "False") in ["1", "True"]
        error_msg = f"AI device for platform '{platform}' is not available. Please check your runtime environment."
        if skip_platform_check:
            logger.warning(error_msg)
            return True
        raise RuntimeError(error_msg)
    logger.info(f"AI device for platform '{platform}' is available.")
    return True
