import random
from typing import Optional

from pydantic import BaseModel, Field

from ..utils.generate_task_id import generate_task_id


def generate_random_seed() -> int:
    return random.randint(0, 2**32 - 1)


class TalkObject(BaseModel):
    audio: str = Field(..., description="Audio path")
    mask: str = Field(..., description="Mask path")


class BaseTaskRequest(BaseModel):
    task_id: str = Field(default_factory=generate_task_id, description="Task ID (auto-generated)")
    prompt: str = Field("", description="Generation prompt")
    use_prompt_enhancer: bool = Field(False, description="Whether to use prompt enhancer")
    negative_prompt: str = Field("", description="Negative prompt")
    image_path: str = Field("", description="Base64 encoded image or URL")
    save_result_path: str = Field("", description="Save result path (optional, defaults to task_id, suffix auto-detected)")
    infer_steps: int = Field(5, description="Inference steps")
    seed: int = Field(default_factory=generate_random_seed, description="Random seed (auto-generated if not set)")
    target_shape: list[int] = Field([], description="Return video or image shape")
    lora_name: Optional[str] = Field(None, description="LoRA filename to load from lora_dir, None to disable LoRA")
    lora_strength: float = Field(1.0, description="LoRA strength")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.save_result_path:
            self.save_result_path = f"{self.task_id}"

    def get(self, key, default=None):
        return getattr(self, key, default)


class VideoTaskRequest(BaseTaskRequest):
    num_fragments: int = Field(1, description="Number of fragments")
    target_video_length: int = Field(81, description="Target video length")
    audio_path: str = Field("", description="Input audio path (Wan-Audio)")
    video_duration: int = Field(5, description="Video duration (Wan-Audio)")
    talk_objects: Optional[list[TalkObject]] = Field(None, description="Talk objects (Wan-Audio)")
    target_fps: Optional[int] = Field(16, description="Target FPS for video frame interpolation (overrides config)")
    resize_mode: Optional[str] = Field("adaptive", description="Resize mode (adaptive, keep_ratio_fixed_area, fixed_min_area, fixed_max_area, fixed_shape, fixed_min_side)")


class ImageTaskRequest(BaseTaskRequest):
    aspect_ratio: str = Field("16:9", description="Output aspect ratio")


class TaskRequest(BaseTaskRequest):
    num_fragments: int = Field(1, description="Number of fragments")
    target_video_length: int = Field(81, description="Target video length (video only)")
    audio_path: str = Field("", description="Input audio path (Wan-Audio)")
    video_duration: int = Field(5, description="Video duration (Wan-Audio)")
    talk_objects: Optional[list[TalkObject]] = Field(None, description="Talk objects (Wan-Audio)")
    aspect_ratio: str = Field("16:9", description="Output aspect ratio (T2I only)")
    target_fps: Optional[int] = Field(16, description="Target FPS for video frame interpolation (overrides config)")


class TaskStatusMessage(BaseModel):
    task_id: str = Field(..., description="Task ID")


class TaskResponse(BaseModel):
    task_id: str
    task_status: str
    save_result_path: str


class StopTaskResponse(BaseModel):
    stop_status: str
    reason: str
