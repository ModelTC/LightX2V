import os
from einops import rearrange
from loguru import logger

import torch
import torchvision
import numpy as np
import imageio
import random

import os
from pydantic import BaseModel, ValidationError


def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def cache_video(
    tensor,
    save_file,
    fps=30,
    suffix=".mp4",
    nrow=8,
    normalize=True,
    value_range=(-1, 1),
    retry=5,
):
    save_dir = os.path.dirname(save_file)
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory: {save_dir}, error: {e}")
        return None

    cache_file = save_file

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack(
                [torchvision.utils.make_grid(u, nrow=nrow, normalize=normalize, value_range=value_range) for u in tensor.unbind(2)],
                dim=1,
            ).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            # write video
            writer = imageio.get_writer(cache_file, fps=fps, codec="libx264", quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            del tensor
            torch.cuda.empty_cache()
            return cache_file
        except Exception as e:
            error = e
            continue
    else:
        logger.info(f"cache_video failed, error: {error}", flush=True)
        return None


def validate_dict(data: dict, model: BaseModel) -> bool:
    try:
        # 使用 Pydantic 的 parse_obj 方法进行验证
        model.parse_obj(data)
        return True  # 如果验证通过，返回 True
    except ValidationError as e:
        # 如果验证失败，打印错误信息并返回 False
        print("Validation Error:", e)
        return False