"""
Wan2.2 distilled model image-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.2 distilled model for I2V generation.
"""

import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads((Path(__file__).resolve().parents[4] / "configs/distill/wan22/wan_moe_i2v_distill_model.json").read_text())
config.update(
    {
        "model_path": "/path/to/wan2.2/Wan2.2-I2V-A14B",
        "model_cls": "wan2.2_moe",
        "task": "i2v",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 42
    prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    save_result_path = "/path/to/save_results/output.mp4"
    image_path = "/path/to/img_0.jpg"

    pipe.generate(
        seed=seed,
        image_path=image_path,
        prompt=prompt,
        save_result_path=save_result_path,
    )
