"""Build mutable inference contexts from request definitions and runtime state."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, make_dataclass
from typing import Any


def _copy_input_containers(value):
    if isinstance(value, dict):
        return {key: _copy_input_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_input_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_input_containers(item) for item in value)
    return value


@dataclass
class InputInfo:
    def update(self, values: Mapping[str, Any], *, copy_nested: bool = False) -> None:
        for input_field in fields(self):
            if input_field.name in values:
                value = values[input_field.name]
                if copy_nested:
                    value = _copy_input_containers(value)
                elif isinstance(value, list):
                    value = value.copy()
                elif isinstance(value, dict):
                    value = dict(value)
                setattr(self, input_field.name, value)


def build_input_info_type(name, arguments, state_type=InputInfo):
    request_fields = [(name, argument.type, argument.input_field()) for name, argument in arguments.items()]
    return make_dataclass(name, request_fields, bases=(state_type,))


@dataclass
class ShapeState(InputInfo):
    latent_shape: list = field(default_factory=list)
    original_shape: list = field(default_factory=list)
    resized_shape: list = field(default_factory=list)


@dataclass
class AudioVideoState(ShapeState):
    video_latent_shape: list = field(default_factory=list)
    audio_latent_shape: list = field(default_factory=list)


@dataclass
class ImageState(ShapeState):
    image_shapes: list = field(default_factory=list)
    txt_seq_lens: list = field(default_factory=list)
    processed_image_size: list = field(default_factory=list)
    original_size: list = field(default_factory=list)
    latent_image_ids: Any = field(default=None, repr=False)
    txt_ids: Any = field(default=None, repr=False)
    revised_prompts: Any = field(default=None, repr=False)
    image_encoder_output: Any = field(default=None, repr=False)
    input_image: Any = field(default=None, repr=False)


@dataclass
class SRState(ShapeState):
    output_fps: float | None = field(default=None, repr=False)


def calculate_num_frames_from_duration(duration_seconds: float, fps: int = 16) -> int:
    """Calculate num_frames from video duration using the formula:
    num_frames = (fps * seconds + 3) // 4 * 4 + 1

    This ensures the result satisfies the VAE stride constraint: (n-1) % 4 == 0

    Args:
        duration_seconds: Video duration in seconds
        fps: Target frames per second (default 16)

    Returns:
        Frame count that satisfies VAE stride constraint

    Examples:
        1s: (16*1 + 3) // 4 * 4 + 1 = 17 frames
        3s: (16*3 + 3) // 4 * 4 + 1 = 49 frames
        5s: (16*5 + 3) // 4 * 4 + 1 = 81 frames
    """
    return align_num_frames(int(fps * duration_seconds) + 3, 4)


def align_num_frames(num_frames: int, temporal_stride: int) -> int:
    """Align a frame count so that ``num_frames - 1`` is stride-divisible."""
    return num_frames // temporal_stride * temporal_stride + 1
