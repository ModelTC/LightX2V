"""LingBot-Video text-to-video inference with a LoRA checkpoint."""

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
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/lingbot_video/lingbot_video_lora.json"

pipe = LightX2VPipeline(
    model_path=MODEL_PATH,
    model_cls="lingbot_video",
    task="t2v",
)

pipe.enable_lora(
    [{"path": LORA_PATH, "strength": 1.0}],
    lora_dynamic_apply=False,
)

pipe.create_generator(config_json=CONFIG_PATH)

prompt = {
    "caption": {
        "comprehensive_description": {
            "scene_content_description": (
                "A sleek white humanoid robot and a fit young man in black athletic wear run side by side along a promenade lined with blooming cherry trees in natural daylight."
            ),
            "camera_movement_description": ("The camera tracks backward smoothly, keeping the runners centered while the background exhibits subtle motion blur."),
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
                "name": "young man",
                "description": "A fit young man wearing black athletic clothes.",
                "actions": [{"timestamp": "[0.0s - 5.0s]", "action": "Runs steadily toward the camera."}],
                "location": "left side of the frame",
                "relative_size": "large",
                "relationship": "Runs beside the humanoid robot.",
            },
            {
                "name": "humanoid robot",
                "description": "A sleek white robot with black mechanical joints.",
                "actions": [{"timestamp": "[0.0s - 5.0s]", "action": "Runs with a fluid gait matching the man's pace."}],
                "location": "right side of the frame",
                "relative_size": "large",
                "relationship": "Runs beside the young man.",
            },
        ],
    },
    "duration": 5,
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
            "incoherent motion",
        ],
    }
}

pipe.generate(
    seed=42,
    prompt=json.dumps(prompt, ensure_ascii=False),
    negative_prompt=json.dumps(negative_prompt, ensure_ascii=False),
    save_result_path="save_results/lingbot_video_t2v_lora.mp4",
)
