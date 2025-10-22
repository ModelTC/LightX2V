import inspect
from dataclasses import dataclass, field


@dataclass
class T2VInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    prompt_enhanced: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    save_result_path: str = field(default_factory=str)
    return_result_tensor: bool = field(default_factory=lambda: False)
    # shape related
    latent_shape: list = field(default_factory=list)
    target_shape: int = field(default_factory=int)


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
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: int = field(default_factory=int)


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
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: int = field(default_factory=int)


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
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: int = field(default_factory=int)


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
    # shape related
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: int = field(default_factory=int)


# Need Check
@dataclass
class AnimateInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    prompt_enhanced: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    image_path: str = field(default_factory=str)
    save_result_path: str = field(default_factory=str)
    return_result_tensor: bool = field(default_factory=lambda: False)
    # shape related
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: int = field(default_factory=int)


@dataclass
class T2IInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    save_result_path: str = field(default_factory=str)
    # shape related
    target_shape: int = field(default_factory=int)


@dataclass
class I2IInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    image_path: str = field(default_factory=str)
    save_result_path: str = field(default_factory=str)
    # shape related
    target_shape: int = field(default_factory=int)
    processed_image_size: int = field(default_factory=list)
    original_size: list = field(default_factory=list)


def set_input_info(args):
    if args.task == "t2v":
        input_info = T2VInputInfo(
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            save_result_path=args.save_result_path,
            return_result_tensor=args.return_result_tensor,
        )
    elif args.task == "i2v":
        input_info = I2VInputInfo(
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image_path=args.image_path,
            save_result_path=args.save_result_path,
            return_result_tensor=args.return_result_tensor,
        )
    elif args.task == "flf2v":
        input_info = Flf2vInputInfo(
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image_path=args.image_path,
            last_frame_path=args.last_frame_path,
            save_result_path=args.save_result_path,
            return_result_tensor=args.return_result_tensor,
        )
    elif args.task == "vace":
        input_info = VaceInputInfo(
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            src_ref_images=args.src_ref_images,
            src_video=args.src_video,
            src_mask=args.src_mask,
            save_result_path=args.save_result_path,
            return_result_tensor=args.return_result_tensor,
        )
    elif args.task == "s2v":
        input_info = S2VInputInfo(
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image_path=args.image_path,
            audio_path=args.audio_path,
            save_result_path=args.save_result_path,
            return_result_tensor=args.return_result_tensor,
        )
    elif args.task == "animate":
        input_info = AnimateInputInfo(
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image_path=args.image_path,
            save_result_path=args.save_result_path,
            return_result_tensor=args.return_result_tensor,
        )
    elif args.task == "t2i":
        input_info = T2IInputInfo(
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            save_result_path=args.save_result_path,
        )
    elif args.task == "i2i":
        input_info = I2IInputInfo(
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image_path=args.image_path,
            save_result_path=args.save_result_path,
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")
    return input_info


def get_all_input_info_keys():
    all_keys = set()

    current_module = inspect.currentframe().f_globals

    for name, obj in current_module.items():
        if inspect.isclass(obj) and name.endswith("InputInfo") and hasattr(obj, "__dataclass_fields__"):
            all_keys.update(obj.__dataclass_fields__.keys())

    return all_keys


# 创建包含所有InputInfo字段的集合
ALL_INPUT_INFO_KEYS = get_all_input_info_keys()
