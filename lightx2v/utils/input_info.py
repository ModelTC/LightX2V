import inspect
from dataclasses import dataclass, field

import torch


@dataclass
class T2VInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    prompt_enhanced: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    save_result_path: str = field(default_factory=str)
    return_result_tensor: bool = field(default_factory=lambda: False)
    # shape related
    resize_mode: str = field(default_factory=str)
    latent_shape: list = field(default_factory=list)
    target_shape: list = field(default_factory=list)


@dataclass
class I2VInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    prompt_enhanced: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    image_path: str = field(default_factory=str)
    save_result_path: str = field(default_factory=str)
    return_result_tensor: bool = field(default_factory=lambda: False)
    # shape related
    resize_mode: str = field(default_factory=str)
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: list = field(default_factory=list)


@dataclass
class Flf2vInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    prompt_enhanced: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    image_path: str = field(default_factory=str)
    last_frame_path: str = field(default_factory=str)
    save_result_path: str = field(default_factory=str)
    return_result_tensor: bool = field(default_factory=lambda: False)
    # shape related
    resize_mode: str = field(default_factory=str)
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: list = field(default_factory=list)


# Need Check
@dataclass
class VaceInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    prompt_enhanced: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    src_ref_images: str = field(default_factory=str)
    src_video: str = field(default_factory=str)
    src_mask: str = field(default_factory=str)
    save_result_path: str = field(default_factory=str)
    return_result_tensor: bool = field(default_factory=lambda: False)
    # shape related
    resize_mode: str = field(default_factory=str)
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: list = field(default_factory=list)


@dataclass
class S2VInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    prompt_enhanced: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    image_path: str = field(default_factory=str)
    audio_path: str = field(default_factory=str)
    audio_num: int = field(default_factory=int)
    with_mask: bool = field(default_factory=lambda: False)
    save_result_path: str = field(default_factory=str)
    return_result_tensor: bool = field(default_factory=lambda: False)
    stream_config: dict = field(default_factory=dict)
    # shape related
    resize_mode: str = field(default_factory=str)
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: list = field(default_factory=list)

    # prev info
    overlap_frame: torch.Tensor = field(default_factory=lambda: None)
    overlap_latent: torch.Tensor = field(default_factory=lambda: None)
    # input preprocess audio
    audio_clip: torch.Tensor = field(default_factory=lambda: None)


# Need Check
@dataclass
class AnimateInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    prompt_enhanced: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    image_path: str = field(default_factory=str)
    src_pose_path: str = field(default_factory=str)
    src_face_path: str = field(default_factory=str)
    src_ref_images: str = field(default_factory=str)
    src_bg_path: str = field(default_factory=str)
    src_mask_path: str = field(default_factory=str)
    save_result_path: str = field(default_factory=str)
    return_result_tensor: bool = field(default_factory=lambda: False)
    # shape related
    resize_mode: str = field(default_factory=str)
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: list = field(default_factory=list)


@dataclass
class T2IInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    save_result_path: str = field(default_factory=str)
    # shape related
    resize_mode: str = field(default_factory=str)
    target_shape: list = field(default_factory=list)
    image_shapes: list = field(default_factory=list)
    txt_seq_lens: list = field(default_factory=list)  # [postive_txt_seq_len, negative_txt_seq_len]
    aspect_ratio: str = field(default_factory=str)


@dataclass
class I2IInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    image_path: str = field(default_factory=str)
    save_result_path: str = field(default_factory=str)
    # shape related
    resize_mode: str = field(default_factory=str)
    target_shape: list = field(default_factory=list)
    image_shapes: list = field(default_factory=list)
    txt_seq_lens: list = field(default_factory=list)  # [postive_txt_seq_len, negative_txt_seq_len]
    processed_image_size: int = field(default_factory=list)
    original_size: list = field(default_factory=list)
    aspect_ratio: str = field(default_factory=str)


def init_empty_input_info(task):
    if task == "t2v":
        return T2VInputInfo()
    elif task == "i2v":
        return I2VInputInfo()
    elif task == "flf2v":
        return Flf2vInputInfo()
    elif task == "vace":
        return VaceInputInfo()
    elif task == "s2v":
        return S2VInputInfo()
    elif task == "animate":
        return AnimateInputInfo()
    elif task == "t2i":
        return T2IInputInfo()
    elif task == "i2i":
        return I2IInputInfo()
    else:
        raise ValueError(f"Unsupported task: {task}")


def update_input_info_from_dict(input_info, data):
    for key in input_info.__dataclass_fields__:
        if key in data:
            setattr(input_info, key, data[key])


def get_all_input_info_keys():
    all_keys = set()

    current_module = inspect.currentframe().f_globals

    for name, obj in current_module.items():
        if inspect.isclass(obj) and name.endswith("InputInfo") and hasattr(obj, "__dataclass_fields__"):
            all_keys.update(obj.__dataclass_fields__.keys())

    return all_keys


# 创建包含所有InputInfo字段的集合
ALL_INPUT_INFO_KEYS = get_all_input_info_keys()
