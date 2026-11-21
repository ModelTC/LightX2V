"""
Wan2.2 image-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.2 model for I2V generation.
"""

import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads((Path(__file__).resolve().parents[3] / "configs/wan22/wan_moe_i2v.json").read_text())
config.update(
    {
        "model_path": "/path/to/Wan2.2-I2V-A14B",
        "model_cls": "wan2.2_moe",
        "task": "i2v",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 42
    prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    image_path = "/path/to/img_0.jpg"
    save_result_path = "/path/to/save_results/output.mp4"

    pipe.generate(
        seed=seed,
        image_path=image_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        save_result_path=save_result_path,
    )
