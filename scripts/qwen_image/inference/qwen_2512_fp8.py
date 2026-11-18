"""
Qwen-image-edit Text-to-image generation example.
This example demonstrates how to use LightX2V with Qwen-Image model for T2I generation.
"""

import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads((Path(__file__).resolve().parents[3] / "configs/qwen_image/qwen_image_t2i_2512_distill_fp8.json").read_text())
config.update(
    {
        "model_path": "/path/to/Qwen/Qwen-Image-2512",
        "model_cls": "qwen_image",
        "task": "t2i",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 42
    prompt = 'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition, Ultra HD, 4K, cinematic composition.'
    save_result_path = "/path/to/save_results/output.png"

    pipe.generate(
        seed=seed,
        prompt=prompt,
        save_result_path=save_result_path,
    )
