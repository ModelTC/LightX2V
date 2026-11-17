"""
Qwen-image-edit image-to-image generation example.
This example demonstrates how to use LightX2V with Qwen-Image-Edit model for I2I generation.
"""

import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads((Path(__file__).resolve().parents[3] / "configs/qwen_image/qwen_image_i2i_2511_distill_lora_fp8.json").read_text())
config.update(
    {
        "model_path": "/path/to/Qwen-Image-Edit-2511",
        "model_cls": "qwen_image",
        "task": "i2i",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 42
    prompt = "Replace the polka-dot shirt with a light blue shirt."
    image_path = ["/path/to/img.png"]  # or ["/path/to/img_0.jpg", "/path/to/img_1.jpg"]
    save_result_path = "/path/to/save_results/output.png"

    pipe.generate(
        seed=seed,
        image_path=image_path,
        prompt=prompt,
        save_result_path=save_result_path,
    )
