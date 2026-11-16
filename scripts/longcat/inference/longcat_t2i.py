"""
LongCat Image text-to-image generation example.
This example demonstrates how to use LightX2V with LongCat-Image model for T2I generation.
"""

import json
from pathlib import Path

from lightx2v import LightX2VPipeline

repo_root = Path(__file__).resolve().parents[3]

config = json.loads((repo_root / "configs/longcat_image/longcat_image_t2i.json").read_text())
config.update(
    {
        "model_path": "/path/to/LongCat-Image",
        "model_cls": "longcat_image",
        "task": "t2i",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 42
    prompt = "一只小狗躺在沙发上"
    negative_prompt = ""
    save_result_path = "/path/to/save_results/longcat_image_t2i_pipeline.png"

    pipe.generate(
        seed=seed,
        prompt=prompt,
        negative_prompt=negative_prompt,
        save_result_path=save_result_path,
    )
