"""
Wan2.1 image-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.1 model for I2V generation.
"""

import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads((Path(__file__).resolve().parents[4] / "configs/wan/wan_i2v_nvfp4.json").read_text())
config.update(
    {
        "model_path": "/path/to/Wan2.1-I2V-14B-480P",
        "model_cls": "wan2.1",
        "task": "i2v",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 42
    prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    image_path = "/path/to/img_0.jpg"
    save_result_path = "/path/to/save_results/output.mp4"

    pipe.generate(
        seed=seed,
        image_path=image_path,
        prompt=prompt,
        save_result_path=save_result_path,
    )
