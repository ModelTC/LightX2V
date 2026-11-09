from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from lightx2v.models.runners.registry import get_model_spec
from lightx2v.utils.request_spec import create_request_model

from ..utils.generate_task_id import generate_task_id


def hide_request_defaults(schema, *, keep_defaults=()):
    """Omitted overrides use runner defaults; retain documented protocol defaults."""
    for name, field_schema in schema["properties"].items():
        if name not in keep_defaults:
            field_schema.pop("default", None)


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
    """HTTP envelope. Only submitted fields become runner overrides.

    Schema defaults initialize this envelope; inference defaults come from the
    runner's startup config and InputInfo, not from omitted HTTP fields.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra=lambda schema: hide_request_defaults(schema))

    task_id: str = Field(default_factory=generate_task_id, description="Task ID (auto-generated)")
    image_mask_path: str = Field("", description="Mask image path (URL, base64, or local path)")
    presigned_url: str = Field("", description="Optional presigned URL for uploading final sync result")
    reuse: bool = Field(False, description="Reuse the previous successful request")
    lora_name: Optional[str] = Field(None, description="LoRA filename to load from lora_dir, None to disable LoRA")
    lora_strength: float = Field(1.0, description="LoRA strength")

    def get(self, key, default=None):
        return getattr(self, key, default)


class VideoTaskRequest(BaseTaskRequest):
    reuse_prefix_segments: int = Field(0, ge=0, description="Number of prefix segments to reuse from the previous successful request")
    talk_objects: Optional[list[TalkObject]] = Field(None, description="Audio and mask inputs for multi-person speech generation")


class ImageTaskRequest(BaseTaskRequest):
    _prefer_memory_result: bool = PrivateAttr(default=False)


def build_task_request(config, base):
    """Build this service's request schema from its model and supported tasks."""
    spec = get_model_spec(config["model_cls"])
    tasks = spec.get_tasks(config)
    single_task = len(tasks) == 1
    models = []
    for task in tasks:
        arguments = {name: argument for name, argument in spec.get_arguments(config, task).items() if argument.server and name != "task"}
        fields = {} if single_task else {"task": (Literal[task], Field(..., description="Inference task"))}
        model = create_request_model(
            f"{config['model_cls']}_{task}_{base.__name__}",
            arguments,
            base=base,
            **fields,
        )
        models.append(model)
    if single_task:
        return models[0]
    return Annotated[Union[tuple(models)], Field(discriminator="task")]


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
