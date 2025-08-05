from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from ..utils.generate_task_id import generate_task_id


class TaskRequest(BaseModel):
    task_id: str = Field(default_factory=generate_task_id, description="Task ID (auto-generated)")
    prompt: str = Field("", description="Generation prompt")
    use_prompt_enhancer: bool = Field(False, description="Whether to use prompt enhancer")
    negative_prompt: str = Field("", description="Negative prompt")
    image_path: str = Field("", description="Input image path")
    num_fragments: int = Field(1, description="Number of fragments")
    save_video_path: str = Field("", description="Save video path (optional, defaults to task_id.mp4)")
    infer_steps: int = Field(5, description="Inference steps")
    target_video_length: int = Field(81, description="Target video length")
    seed: int = Field(42, description="Random seed")
    audio_path: str = Field("", description="Input audio path (Wan-Audio)")
    video_duration: int = Field(5, description="Video duration (Wan-Audio)")

    def __init__(self, **data):
        super().__init__(**data)
        # If save_video_path is empty, use task_id.mp4
        if not self.save_video_path:
            self.save_video_path = f"{self.task_id}.mp4"

    def get(self, key, default=None):
        return getattr(self, key, default)


class TaskStatusMessage(BaseModel):
    task_id: str = Field(..., description="Task ID")


class TaskResponse(BaseModel):
    task_id: str
    task_status: str
    save_video_path: str


class TaskResultResponse(BaseModel):
    status: str
    task_status: str
    filename: Optional[str] = None
    file_size: Optional[int] = None
    download_url: Optional[str] = None
    message: str


class ServiceStatusResponse(BaseModel):
    service_status: str
    task_id: Optional[str] = None
    start_time: Optional[datetime] = None


class StopTaskResponse(BaseModel):
    stop_status: str
    reason: str
