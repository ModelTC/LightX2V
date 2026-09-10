from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ..utils.generate_task_id import generate_task_id


class UsageInputTokensDetails(BaseModel):
    image_tokens: int = 0
    text_tokens: int = 0


class UsageOutputTokensDetails(BaseModel):
    image_tokens: int = 0
    text_tokens: int = 0


class Usage(BaseModel):
    input_tokens: int
    input_tokens_details: UsageInputTokensDetails
    output_tokens: int
    total_tokens: int
    output_tokens_details: UsageOutputTokensDetails


class TalkObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: str = Field(..., description="Audio path")
    mask: str = Field(..., description="Mask path")


class BaseTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(default_factory=generate_task_id, description="Task ID (auto-generated)")
    task: Optional[str] = Field(None, description="Required for multi-task runners; single-task services use their only supported task")
    prompt: str = Field("", description="Generation prompt")
    negative_prompt: str = Field("", description="Negative prompt")
    image_path: str = Field("", description="Base64 encoded image or URL")
    last_frame_path: str = Field("", description="Last frame image path (base64, or local path)")
    image_mask_path: str = Field("", description="Mask image path (supports URL, base64, or local path)")
    save_result_path: Optional[str] = Field(None, description="Output path; omitted or null skips file saving")
    presigned_url: str = Field("", description="Optional presigned URL for uploading final sync result")
    seed: Optional[int] = Field(None, description="Non-negative seed; omitted or null defaults to 42")
    reuse: bool = Field(False, description="Reuse the previous successful request")
    size: list[int] = Field([], description="Output size in pixels: [height, width]")
    lora_name: Optional[str] = Field(None, description="LoRA filename to load from lora_dir, None to disable LoRA")
    lora_strength: float = Field(1.0, description="LoRA strength")

    def get(self, key, default=None):
        return getattr(self, key, default)


class VideoTaskRequest(BaseTaskRequest):
    num_frames: Optional[int] = Field(
        None,
        description="Number of output frames; defaults to the startup config",
    )
    reuse_prefix_segments: int = Field(
        0,
        ge=0,
        description="Number of prefix segments to reuse from the previous successful request",
    )
    video_path: str = Field("", description="Server-local input video path; comma-separated paths for multiple references")
    sr_ratio: float = Field(2.0, gt=0, description="Super-resolution scale factor used when size is not set")
    audio_path: str = Field("", description="Input audio path (Wan-Audio)")
    video_duration: float = Field(5, description="Video duration in seconds (Wan-Audio)")
    talk_objects: Optional[list[TalkObject]] = Field(None, description="Talk objects (Wan-Audio)")
    ref_image_paths: list[str] = Field(default_factory=list, description="VACE reference images as base64, URL, or server-local paths")
    ref_video_prompt: Optional[str] = Field(None, description="Prompt describing the driving video for Wan-Animate-2")
    pose_video_path: Optional[str] = Field(None, description="Server-local pose video path")
    face_video_path: Optional[str] = Field(None, description="Server-local face video path")
    background_video_path: Optional[str] = Field(None, description="Server-local background video path")
    mask_path: Optional[str] = Field(None, description="Server-local mask video path for VACE or animation replacement")
    pose: str | dict[str, Any] | None = Field(None, description="Action string, server-local pose JSON path, or WorldPlay pose object")
    action_path: Optional[str] = Field(None, description="Server-local action file or control directory")
    state_path: Optional[str] = Field(None, description="Server-local robot state file")
    action_mode: Optional[str] = Field(None, description="Cosmos3 action mode")
    domain_name: Optional[str] = Field(None, description="Cosmos3 embodiment domain")
    view_point: Optional[str] = Field(None, description="Cosmos3 viewpoint")
    save_action_path: Optional[str] = Field(None, description="Action output path, resolved by the runner")
    image_strength: float | list[float] | None = Field(None, description="LTX image conditioning strength, shared or per image")
    image_frame_indices: Optional[list[int]] = Field(None, description="LTX pixel frame index for each conditioning image")
    reference_video_strength: Optional[float] = Field(None, description="LTX reference-video conditioning strength")
    reference_video_frame_cap: Optional[int] = Field(None, description="Maximum number of LTX reference-video frames")
    mux_audio_video_path: Optional[str] = Field(None, description="Server-local media file whose audio is muxed into the saved LTX video")


class ImageTaskRequest(BaseTaskRequest):
    # Sync image APIs return PNG bytes without requiring an output file.
    _prefer_memory_result: bool = PrivateAttr(default=False)

    aspect_ratio: str = Field("16:9", description="Output aspect ratio")
    i2i_denoise_strength: Optional[float] = Field(None, description="Single-image I2I edit denoising strength in [0.0, 1.0]; omit to keep existing behavior")
    inpaint_blur_sigma: Optional[float] = Field(None, description="Flux2 inpainting mask blur sigma")
    inpaint_blur_size: Optional[int] = Field(None, description="Flux2 inpainting mask blur kernel size")
    sr_ratio: float = Field(2.0, gt=0, description="Super-resolution scale factor used when size is not set")
    keep_aspect_ratio: Optional[bool] = Field(None, description="Preserve a single HiDream reference image's aspect ratio")
    layout_bboxes: Optional[str] = Field(None, description="HiDream layout boxes as a JSON string or server-local JSON file path")
    align_image_size: Optional[bool] = Field(None, description="Align HunyuanImage3 reference image sizes during inference")


class SenseNovaVisionTaskRequest(BaseModel):
    """One request for the multi-task SenseNova-Vision service."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(default_factory=generate_task_id, description="Task ID (auto-generated)")
    task: str = Field(..., description="Public SenseNova-Vision task name")
    prompt: str = Field("", description="Task prompt or question")
    images: list[str] = Field(
        default_factory=list,
        description="Input images as base64/data URLs, HTTP(S) URLs, or server-local paths",
    )
    seed: Optional[int] = Field(None, description="Non-negative seed; omitted or null defaults to 42")
    visualize: bool = Field(True, description="Generate an official-style visualization when supported")
    postprocess_3d: bool = Field(False, description="Generate a GLB scene for recon3d")

    def get(self, key, default=None):
        return getattr(self, key, default)


class SenseNovaArtifact(BaseModel):
    kind: str
    media_type: str
    filename: str
    url: str
    size_bytes: int


class SenseNovaVisionTaskSubmission(BaseModel):
    task_id: str
    task_status: str
    task: str


class SenseNovaVisionTaskResult(BaseModel):
    task_id: str
    status: str
    task: str
    runner_task: str
    mode: str
    text: Optional[str] = None
    artifacts: list[SenseNovaArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SenseNovaVisionGenerationResponse(BaseModel):
    """Internal response passed from the generation service to TaskManager."""

    task_id: str
    task_status: str
    save_result_path: str = ""
    result_data: dict[str, Any]


class TaskStatusMessage(BaseModel):
    task_id: str = Field(..., description="Task ID")


class TaskResponse(BaseModel):
    task_id: str
    task_status: str
    save_result_path: Optional[str]
    # Filled after image generation in-process; never serialized in JSON responses.
    result_png: Optional[bytes] = Field(default=None, exclude=True)
    usage: Optional[Usage] = Field(default=None, exclude=True)


class StopTaskResponse(BaseModel):
    stop_status: str
    reason: str
