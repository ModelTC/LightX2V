from dataclasses import dataclass, field


@dataclass
class T2VInputInfo:
    seed: int = field(default_factory=int)
    prompt: str = field(default_factory=str)
    prompt_enhanced: str = field(default_factory=str)
    negative_prompt: str = field(default_factory=str)
    save_result_path: str = field(default_factory=str)
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
    # shape related
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: int = field(default_factory=int)


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
    save_result_path: str = field(default_factory=str)
    # shape related
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)
    latent_shape: list = field(default_factory=list)
    target_shape: int = field(default_factory=int)
