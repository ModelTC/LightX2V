"""Request fields for LightX2V-native StreamRender inference."""

from dataclasses import dataclass, field

from lightx2v.utils.input_info import InputInfo


@dataclass
class StreamRenderH3InputInfo(InputInfo):
    prompt: str = ""
    image_path: str = ""
    video_path: str = ""
    num_frames: int | None = None
    size: list[int] = field(default_factory=list)
