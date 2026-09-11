"""Reusable request field groups; each runner declares its task-specific fields."""

COMMON_REQUEST_FIELDS = frozenset({"return_result_tensor", "save_result_path", "seed", "task"})
PROMPT_FIELDS = frozenset({"negative_prompt", "prompt"})
VIDEO_OUTPUT_FIELDS = frozenset({"size", "num_frames"})

IMAGE_REQUEST_FIELDS = COMMON_REQUEST_FIELDS | PROMPT_FIELDS | {"aspect_ratio", "size"}
VIDEO_REQUEST_FIELDS = COMMON_REQUEST_FIELDS | PROMPT_FIELDS | VIDEO_OUTPUT_FIELDS
