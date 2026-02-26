import inspect
from dataclasses import dataclass, field, fields
from typing import Any

import torch


class _UnsetType:
    def __repr__(self):
        return "UNSET"


UNSET = _UnsetType()


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
    # WorldPlay-specific: pose/action conditioning (optional)
    pose: str = field(default_factory=lambda: None)


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


@dataclass
class RS2VInputInfo:
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
    # input reference state
    ref_state: int = field(default_factory=int)
    # flags for first and last clip
    is_first: bool = field(default_factory=lambda: False)
    is_last: bool = field(default_factory=lambda: False)


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
    return_result_tensor: bool = field(default_factory=lambda: False)
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
    return_result_tensor: bool = field(default_factory=lambda: False)
    # shape related
    resize_mode: str = field(default_factory=str)
    target_shape: list = field(default_factory=list)
    image_shapes: list = field(default_factory=list)
    txt_seq_lens: list = field(default_factory=list)  # [postive_txt_seq_len, negative_txt_seq_len]
    processed_image_size: int = field(default_factory=list)
    original_size: list = field(default_factory=list)
    aspect_ratio: str = field(default_factory=str)


@dataclass
class T2AVInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    prompt_enhanced: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    save_result_path: str = field(default_factory=str)
    return_result_tensor: bool = field(default_factory=lambda: False)
    # shape related
    resize_mode: str = field(default_factory=str)
    audio_latent_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: list = field(default_factory=list)


@dataclass
class I2AVInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    prompt_enhanced: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    image_path: str = field(default_factory=str)
    image_strength: float = field(default_factory=float)
    save_result_path: str = field(default_factory=str)
    return_result_tensor: bool = field(default_factory=lambda: False)
    # shape related
    resize_mode: str = field(default_factory=str)
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: list = field(default_factory=list)


@dataclass
class WorldPlayI2VInputInfo:
    """Input info for WorldPlay model (image-to-video with action/pose conditioning)."""

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
    # WorldPlay-specific: pose/action conditioning
    pose: str = field(default_factory=str)  # Pose string (e.g., "w-3, right-0.5") or JSON path
    model_type: str = field(default_factory=lambda: "ar")  # "ar" (autoregressive) or "bi" (bidirectional)
    chunk_latent_frames: int = field(default_factory=lambda: 4)
    # Computed pose tensors (set during processing)
    viewmats: torch.Tensor = field(default_factory=lambda: None)
    Ks: torch.Tensor = field(default_factory=lambda: None)
    action: torch.Tensor = field(default_factory=lambda: None)


@dataclass
class WorldPlayT2VInputInfo:
    """Input info for WorldPlay model (text-to-video with action/pose conditioning)."""

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
    # WorldPlay-specific: pose/action conditioning
    pose: str = field(default_factory=str)  # Pose string (e.g., "w-3, right-0.5") or JSON path
    model_type: str = field(default_factory=lambda: "ar")  # "ar" (autoregressive) or "bi" (bidirectional)
    chunk_latent_frames: int = field(default_factory=lambda: 4)
    # Computed pose tensors (set during processing)
    viewmats: torch.Tensor = field(default_factory=lambda: None)
    Ks: torch.Tensor = field(default_factory=lambda: None)
    action: torch.Tensor = field(default_factory=lambda: None)


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
    elif task == "rs2v":
        return RS2VInputInfo()
    elif task == "animate":
        return AnimateInputInfo()
    elif task == "t2i":
        return T2IInputInfo()
    elif task == "i2i":
        return I2IInputInfo()
    elif task == "t2av":
        return T2AVInputInfo()
    elif task == "i2av":
        return I2AVInputInfo()
    elif task == "worldplay_i2v":
        return WorldPlayI2VInputInfo()
    elif task == "worldplay_t2v":
        return WorldPlayT2VInputInfo()
    else:
        raise ValueError(f"Unsupported task: {task}")


@dataclass
class SekoTalkInputs:
    infer_steps: int | Any = UNSET
    seed: int | Any = UNSET
    prompt: str | Any = UNSET
    prompt_enhanced: str | Any = UNSET
    negative_prompt: str | Any = UNSET
    image_path: str | Any = UNSET
    audio_path: str | Any = UNSET
    audio_num: int | Any = UNSET
    video_duration: float | Any = UNSET
    with_mask: bool | Any = UNSET
    save_result_path: str | Any = UNSET
    return_result_tensor: bool | Any = UNSET
    stream_config: dict | Any = UNSET

    resize_mode: str | Any = UNSET
    target_shape: list | Any = UNSET

    # prev info
    overlap_frame: torch.Tensor | Any = UNSET
    overlap_latent: torch.Tensor | Any = UNSET
    # input preprocess audio
    audio_clip: torch.Tensor | Any = UNSET

    # input reference state
    ref_state: int | Any = UNSET
    # flags for first and last clip
    is_first: bool | Any = UNSET
    is_last: bool | Any = UNSET

    @classmethod
    def from_args(cls, args, **overrides):
        """
        Build InputInfo from argparse.Namespace (or any object with __dict__)
        Priority:
            args < overrides
        """
        field_names = {f.name for f in fields(cls)}
        data = {k: v for k, v in vars(args).items() if k in field_names}
        data.update(overrides)
        return cls(**data)

    def normalize_unset_to_none(self):
        """
        Replace all UNSET fields with None.
        Call this right before running / inference.
        """
        for f in fields(self):
            if getattr(self, f.name) is UNSET:
                setattr(self, f.name, None)
        return self


def init_input_info_from_args(task, args, **overrides):
    if task in ["s2v", "rs2v"]:
        return SekoTalkInputs.from_args(args, **overrides)
    else:
        raise ValueError(f"Unsupported task: {task}")


def fill_input_info_from_defaults(input_info, defaults):
    for key in input_info.__dataclass_fields__:
        if key in defaults and getattr(input_info, key) is UNSET:
            setattr(input_info, key, defaults[key])


def update_input_info_from_dict(input_info, data):
    for key in input_info.__dataclass_fields__:
        if key in data:
            setattr(input_info, key, data[key])


def update_input_info_from_object(input_info, obj):
    for key in input_info.__dataclass_fields__:
        if hasattr(obj, key):
            setattr(input_info, key, getattr(obj, key))


def get_all_input_info_keys():
    all_keys = set()

    current_module = inspect.currentframe().f_globals

    for name, obj in current_module.items():
        if inspect.isclass(obj) and name.endswith("InputInfo") and hasattr(obj, "__dataclass_fields__"):
            all_keys.update(obj.__dataclass_fields__.keys())

    return all_keys


# 创建包含所有InputInfo字段的集合
ALL_INPUT_INFO_KEYS = get_all_input_info_keys()
