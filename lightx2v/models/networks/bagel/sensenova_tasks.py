# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from typing import Optional

MODE_PROFILES = {
    "generate": {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.4, 1.0],
        "timestep_shift": 3.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "global",
    },
    "think_generate": {
        "max_think_token_n": 1000,
        "do_sample": False,
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.4, 1.0],
        "timestep_shift": 3.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "global",
        "think": True,
    },
    "caption_generate": {
        "max_think_token_n": 8192,
        "do_sample": False,
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 4.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "global",
        "caption": True,
    },
    "dense_perception": {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 4.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "text_channel",
    },
    "edit": {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 2.0,
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 4.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "text_channel",
    },
    "think_edit": {
        "max_think_token_n": 1000,
        "do_sample": False,
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 2.0,
        "cfg_interval": [0.4, 1.0],
        "timestep_shift": 3.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 0.0,
        "cfg_renorm_type": "text_channel",
        "think": True,
    },
    "understanding": {
        "max_think_token_n": 8192,
        "do_sample": False,
        "understanding_output": True,
    },
    "think_understanding": {
        "max_think_token_n": 8192,
        "do_sample": False,
        "understanding_output": True,
        "think": True,
    },
    "dense_detection": {
        "max_think_token_n": 8192,
        "do_sample": False,
        "understanding_output": True,
    },
    "dense_OCR": {
        "max_think_token_n": 20000,
        "do_sample": False,
        "understanding_output": True,
    },
    "recon3d": {
        "cfg_text_scale": 1.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 4.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "text_channel",
    },
}

IMAGE_OUTPUT_MODES = {
    "generate",
    "think_generate",
    "caption_generate",
    "dense_perception",
    "edit",
    "think_edit",
    "recon3d",
}
TEXT_OUTPUT_MODES = {
    "understanding",
    "think_understanding",
    "dense_detection",
    "dense_OCR",
}


@dataclass(frozen=True)
class OmniVisionTaskSpec:
    """Public omni-vision subtask mapped to the existing SenseNova runner path."""

    runner_task: str
    mode: str
    min_images: int = 1
    max_images: int = 1
    requires_prompt: bool = False
    visualizer: Optional[str] = None


# Keep the public task vocabulary independent of the internal runner branches.
# Both offline inference and the resident server consume this single table.
OMNI_VISION_TASK_SPECS = {
    "understanding": OmniVisionTaskSpec("raw_query", "understanding", max_images=10),
    "binary_segmentation": OmniVisionTaskSpec("binary_seg", "dense_perception", requires_prompt=True, visualizer="binary"),
    "depth": OmniVisionTaskSpec("depth", "dense_perception"),
    "normal": OmniVisionTaskSpec("normal", "dense_perception"),
    "gcg_segmentation": OmniVisionTaskSpec("gcg_seg", "caption_generate", visualizer="gcg"),
    "object_detection": OmniVisionTaskSpec("bbox_detection", "understanding", requires_prompt=True, visualizer="detection"),
    "point_detection": OmniVisionTaskSpec("point_detection", "dense_detection", requires_prompt=True, visualizer="detection"),
    "keypoint": OmniVisionTaskSpec("keypoint", "dense_detection", requires_prompt=True, visualizer="detection"),
    "ocr": OmniVisionTaskSpec("ocr", "dense_OCR", visualizer="detection"),
    "recon3d": OmniVisionTaskSpec("recon3d", "recon3d", max_images=10),
    "panoptic_segmentation": OmniVisionTaskSpec("pan_seg", "caption_generate", visualizer="panoptic"),
    "interactive_segmentation": OmniVisionTaskSpec(
        "binary_seg",
        "dense_perception",
        min_images=2,
        max_images=2,
        requires_prompt=True,
        visualizer="interactive",
    ),
    "vgd_segmentation": OmniVisionTaskSpec("gcg_seg", "caption_generate", requires_prompt=True, visualizer="gcg"),
    "camera_pose": OmniVisionTaskSpec("camera_pose", "understanding", min_images=2, max_images=10),
}

OMNI_VISION_SUBTASK_ALIASES = {
    "raw_query": "understanding",
    "binary_seg": "binary_segmentation",
    "gcg_seg": "gcg_segmentation",
    "bbox_detection": "object_detection",
    "pan_seg": "panoptic_segmentation",
    "interactive_seg": "interactive_segmentation",
    "vgd_seg": "vgd_segmentation",
}

OMNI_VISION_SUBTASKS = tuple(OMNI_VISION_TASK_SPECS)
OMNI_VISION_SUBTASK_CHOICES = tuple((*OMNI_VISION_SUBTASKS, *OMNI_VISION_SUBTASK_ALIASES))


def normalize_omni_vision_subtask(subtask):
    normalized = str(subtask or "").strip().lower().replace("-", "_")
    normalized = OMNI_VISION_SUBTASK_ALIASES.get(normalized, normalized)
    if normalized not in OMNI_VISION_TASK_SPECS:
        supported = ", ".join(sorted(OMNI_VISION_TASK_SPECS))
        raise ValueError(f"Unsupported omni-vision subtask {subtask!r}; supported subtasks: {supported}")
    return normalized


def get_omni_vision_task_spec(subtask):
    normalized = normalize_omni_vision_subtask(subtask)
    return normalized, OMNI_VISION_TASK_SPECS[normalized]


TASK_TO_MODE = {
    "raw_query": "dense_perception",
    "depth": "dense_perception",
    "normal": "dense_perception",
    "binary_seg": "dense_perception",
    "pan_seg": "caption_generate",
    "gcg_seg": "caption_generate",
    "bbox_detection": "dense_detection",
    "point_detection": "dense_detection",
    "keypoint": "dense_detection",
    "ocr": "dense_OCR",
    "recon3d": "recon3d",
    "camera_pose": "understanding",
}

DEPTH_PROMPT = (
    "Estimate relative depth for each pixel in the image, with closer objects "
    "appearing brighter and distant objects appearing darker. Output is a "
    "grayscale image with pixel values ranging from 0-255."
)
NORMAL_PROMPT = (
    "Generate an RGB normal map where R, G, B channels represent X, Y, Z surface directions. The output should show continuous color variations with no discrete regions, unlike segmentation results."
)
GCG_PROMPT = "Please briefly describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer."
OCR_PROMPT = (
    "Perform word-level text detection and recognition on the entire image. "
    "Output a structured text list containing every detected word, its bounding "
    "box coordinates with <bbox> format, and the recognized text content."
)
RECON3D_PROMPT = "Reconstruct a scene from multiple input images and output one dense 3D coordinate map per view, all aligned to the first camera's perspective."
CAMERA_POSE_PROMPT = (
    "With the first frame as the reference frame, output the relative pose of"
    " all subsequent frames (excluding the first frame) with respect to the"
    " first frame, following the input order and adhering to the strict format"
    " below:Rotation: Represented by a quaternion in the format"
    " <quat>[x,y,z,w], enclosed in <quat> tags;Translation: Represented by a"
    " unit vector (direction) in the format <offset>[x,y,z], enclosed in"
    " <offset> tags (the vector has no absolute physical meaning, only"
    " directional information);Scale: Represented by a numerical value in the"
    " format <scale>value</scale> tags, where the value denotes the magnitude"
    " of translation (corresponding to the length of the translation unit"
    " vector);Enclose the result of each frame in <frame> tags, with no extra"
    " characters, spaces, or line breaks outside the tags."
)


def resolve_mode(task, requested_mode=""):
    mode = requested_mode.strip() if requested_mode else ""
    if not mode:
        mode = TASK_TO_MODE.get(task)
    if mode not in MODE_PROFILES:
        raise ValueError(f"Unsupported SenseNova-Vision mode {mode!r}; available: {sorted(MODE_PROFILES)}")
    return mode


def get_mode_profile(mode):
    return dict(MODE_PROFILES[mode])


def _format_categories(query):
    query = query.strip()
    if "<p>" in query and "</p>" in query:
        return query
    categories = [item.strip() for item in query.split(",") if item.strip()]
    return ", ".join(f"<p>{item}</p>" for item in categories)


def resolve_prompt(task, prompt):
    prompt = str(prompt or "").strip()
    if task == "depth":
        return prompt or DEPTH_PROMPT
    if task == "normal":
        return prompt or NORMAL_PROMPT
    if task == "gcg_seg":
        return prompt or GCG_PROMPT
    if task == "ocr":
        return prompt or OCR_PROMPT
    if task == "recon3d":
        return prompt or RECON3D_PROMPT
    if task == "camera_pose":
        return prompt or CAMERA_POSE_PROMPT
    if not prompt:
        raise ValueError(f"SenseNova-Vision task={task!r} requires --prompt.")

    if task == "binary_seg" and "segmentation mask" not in prompt.lower():
        categories = _format_categories(prompt)
        return f"Can you segment the image based on the following categories: {categories}? Please output the binary segmentation masks."
    if task == "bbox_detection" and "detect" not in prompt.lower():
        categories = _format_categories(prompt)
        return f"Detect all instances of {categories} in the image. Output the results as a structured text list with each detection including category and bounding box coordinates in <bbox> format."
    if task == "point_detection" and "locate" not in prompt.lower():
        categories = _format_categories(prompt)
        return (
            f"Locate and identify {categories} within the scene. Output detection results as text entries, each containing the object class and pixel coordinates defining the object point location."
        )
    return prompt


def ensure_image_placeholders(prompt, image_count):
    count = prompt.count("<image>")
    if count == image_count:
        return prompt
    if count == 0:
        return ("<image>" * image_count) + prompt
    raise ValueError(f"The number of <image> placeholders ({count}) must match input images ({image_count}).")


def clean_text_output(text):
    return re.sub(r"^\s+|\s+$", "", str(text or ""))
