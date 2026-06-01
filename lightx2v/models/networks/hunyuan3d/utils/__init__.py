"""Shared helpers for vendored Hunyuan3D shape code."""

import os
from functools import wraps

import torch
from loguru import logger

from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def get_logger(name):
    return logger.bind(name=name)


logger = logger.bind(name="hy3dgen.shapgen")


class synchronize_timer:
    """Synchronized timer to count the inference time of `nn.Module.forward`.

    Supports both context manager and decorator usage.

    Example as context manager:
    ```python
    with synchronize_timer('name') as t:
        run()
    ```

    Example as decorator:
    ```python
    @synchronize_timer('Export to trimesh')
    def export_to_trimesh(mesh_output):
        pass
    ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        """Context manager entry: start timing."""
        if os.environ.get("HY3DGEN_DEBUG", "0") == "1":
            self.start = torch_device_module.Event(enable_timing=True)
            self.end = torch_device_module.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Context manager exit: stop timing and log results."""
        if os.environ.get("HY3DGEN_DEBUG", "0") == "1":
            self.end.record()
            torch_device_module.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                logger.info(f"{self.name} takes {self.time} ms")

    def __call__(self, func):
        """Decorator: wrap the function to time its execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper


def smart_load_model(
    model_path,
    subfolder,
    use_safetensors,
    variant,
):
    original_model_path = model_path
    # try local path
    base_dir = os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen")
    model_fld = os.path.expanduser(os.path.join(base_dir, model_path))
    model_path = os.path.expanduser(os.path.join(base_dir, model_path, subfolder))
    logger.info(f"Try to load model from local path: {model_path}")
    if not os.path.exists(model_path):
        logger.info("Model path not exists, try to download from huggingface")
        try:
            from huggingface_hub import snapshot_download

            # 只下载指定子目录
            path = snapshot_download(
                repo_id=original_model_path,
                allow_patterns=[f"{subfolder}/*"],  # 关键修改：模式匹配子文件夹
                local_dir=model_fld,
            )
            model_path = os.path.join(path, subfolder)  # 保持路径拼接逻辑不变
        except ImportError:
            logger.warning("You need to install HuggingFace Hub to load models from the hub.")
            raise RuntimeError(f"Model path {model_path} not found")
        except Exception as e:
            raise e

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {original_model_path} not found")

    extension = "ckpt" if not use_safetensors else "safetensors"
    variant = "" if variant is None else f".{variant}"
    ckpt_name = f"model{variant}.{extension}"
    config_path = os.path.join(model_path, "config.yaml")
    ckpt_path = os.path.join(model_path, ckpt_name)
    return config_path, ckpt_path


__all__ = ["get_logger", "logger", "smart_load_model", "synchronize_timer"]
