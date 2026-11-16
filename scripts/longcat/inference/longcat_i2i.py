"""
LongCat Image Edit (I2I) generation example.
This example demonstrates how to use LightX2V with LongCat-Image-Edit model for I2I generation.
"""

import json
from pathlib import Path

from lightx2v import LightX2VPipeline

repo_root = Path(__file__).resolve().parents[3]

config = json.loads((repo_root / "configs/longcat_image/longcat_image_i2i.json").read_text())
config.update(
    {
        "model_path": "/path/to/LongCat-Image-Edit",
        "model_cls": "longcat_image",
        "task": "i2i",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 43
    prompt = "将猫变成狗"
    negative_prompt = ""
    image_path = "/path/to/LongCat-Image-Edit/assets/test.png"
    save_result_path = "/path/to/save_results/longcat_image_i2i_pipeline.png"

    # Generate edited image
    pipe.generate(
        seed=seed,
        image_path=image_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        save_result_path=save_result_path,
    )
