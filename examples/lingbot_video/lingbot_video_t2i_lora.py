"""LingBot-Video text-to-image inference with a LoRA checkpoint."""

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
    task="t2i",
)

pipe.enable_lora(
    [{"path": LORA_PATH, "strength": 1.0}],
    lora_dynamic_apply=False,
)

pipe.create_generator(config_json=CONFIG_PATH)

prompt = {
    "comprehensive_description": (
        "A clear glass bottle filled with water sits on a light wooden table in "
        "warm sunlight. The bottle focuses the sunlight into a bright starburst "
        "on the table, with softly blurred green foliage in the background."
    ),
    "camera_info": {
        "color": "Warm",
        "frame_size": "Extreme Close Up",
        "shot_type_angle": "Low angle",
        "lens_size": "Long Lens",
        "composition": "Left heavy",
        "lighting": "Hard light",
        "lighting_type": "Daylight",
    },
    "world_knowledge": [],
    "prominent_elements": [
        {
            "name": "glass bottle",
            "description": "A clear cylindrical bottle partially filled with water.",
            "location": "center-left",
            "relative_size": "medium",
            "shape_and_color": "Cylindrical, transparent and clear",
            "texture": "smooth, glossy",
            "relationship": "Focuses sunlight into a bright flare on the table.",
            "orientation": "upright",
        }
    ],
}
negative_prompt = {
    "universal_negative": {
        "visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "underexposed", "overexposed"],
        "artistic_style": ["painting", "illustration", "drawing", "cartoon", "cgi"],
        "composition_and_content": ["text", "watermark", "signature", "logo"],
        "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "distorted reflections"],
    }
}

pipe.generate(
    seed=42,
    prompt=json.dumps(prompt, ensure_ascii=False),
    negative_prompt=json.dumps(negative_prompt, ensure_ascii=False),
    save_result_path="./save_results/lingbot_video_t2i_lora.png",
)
