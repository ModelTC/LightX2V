"""LingBot-Video image-to-video inference with a LoRA checkpoint."""

import json
import os
from pathlib import Path

# Keep the numerical and kernel settings aligned with scripts/lingbot_video.
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["DTYPE"] = "BF16"
os.environ["SENSITIVE_LAYER_DTYPE"] = "FP32"
os.environ["LINGBOT_QWEN_ATTN_IMPLEMENTATION"] = "sdpa"
os.environ["LINGBOT_MOE_PAD_BACKEND"] = "vectorized"
os.environ["LINGBOT_MOE_EXPERT_BACKEND"] = "grouped_mm"

from lightx2v import LightX2VPipeline

MODEL_PATH = "/data/nvme5/gushiqiao/models/lingbot-video-moe-30b-a3b"
LORA_PATH = "/data/nvme6/gushiqiao/models/lingbot_video_4steps_rank128_v2.safetensors"
IMAGE_PATH = "/path/to/input.png"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/lingbot_video/lingbot_video_lora.json"

pipe = LightX2VPipeline(
    model_path=MODEL_PATH,
    model_cls="lingbot_video",
    task="i2v",
)

pipe.enable_lora(
    [{"path": LORA_PATH, "strength": 1.0}],
    lora_dynamic_apply=False,
)

pipe.create_generator(config_json=CONFIG_PATH)

prompt = {
    "comprehensive_description": {
        "scene_content_description": (
            "The subjects from the input image begin running forward together. Their motion is natural and coordinated while their appearance and the original scene composition remain consistent."
        ),
        "camera_movement_description": ("The camera tracks backward smoothly to keep the subjects centered, with subtle background motion blur and stable cinematic lighting."),
    },
    "camera_info": {
        "color": "Natural",
        "frame_size": "Wide",
        "shot_type_angle": "Eye level",
        "lens_size": "Telephoto",
        "composition": "Symmetrical",
        "lighting": "Bright sunlight",
        "lighting_type": "Daylight",
    },
    "world_knowledge": [],
    "prominent_elements": [
        {
            "name": "subjects from the input image",
            "description": "The main subjects shown in the conditioning image.",
            "actions": [
                {
                    "timestamp": "[0.0s - 5.0s]",
                    "action": "Run forward together with natural coordinated movement.",
                }
            ],
            "location": "center of the frame",
            "relative_size": "large",
            "relationship": "Preserve their identity and appearance.",
        }
    ],
}
negative_prompt = {
    "universal_negative": {
        "visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "unstable color"],
        "artistic_style": ["painting", "illustration", "drawing", "cartoon", "cgi"],
        "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles"],
        "temporal_and_motion_stability": [
            "flickering",
            "jittery",
            "motion blur",
            "temporal inconsistency",
            "warping",
            "morphing",
            "inconsistent identity",
        ],
    }
}

pipe.generate(
    seed=42,
    image_path=IMAGE_PATH,
    prompt=json.dumps(prompt, ensure_ascii=False),
    negative_prompt=json.dumps(negative_prompt, ensure_ascii=False),
    save_result_path="save_results/lingbot_video_i2v_lora.mp4",
)
