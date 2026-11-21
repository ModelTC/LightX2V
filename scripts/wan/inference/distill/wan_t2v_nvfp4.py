"""
Wan2.1 text-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.1 model for T2V generation.
"""

import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads((Path(__file__).resolve().parents[4] / "configs/wan/wan_t2v_nvfp4.json").read_text())
config.update(
    {
        "model_path": "/path/to/Wan2.1-T2V-1.3B",
        "model_cls": "wan2.1",
        "task": "t2v",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 42
    prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    save_result_path = "/path/to/save_results/output.mp4"

    pipe.generate(
        seed=seed,
        prompt=prompt,
        save_result_path=save_result_path,
    )
