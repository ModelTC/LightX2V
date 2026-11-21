"""
HunyuanVideo-1.5 image-to-video generation example with quantization.
This example demonstrates how to use LightX2V with HunyuanVideo-1.5 model for I2V generation,
including quantized model usage for reduced memory consumption.
"""

import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads((Path(__file__).resolve().parents[3] / "configs/hunyuan_video_15/quant/hy15_i2v_720p_fp8.json").read_text())
config.update(
    {
        "model_path": "/path/to/ckpts/hunyuanvideo-1.5/",
        "model_cls": "hunyuan_video_1.5",
        "task": "i2v",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 42
    prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    negative_prompt = ""
    save_result_path = "/path/to/save_results/output2.mp4"

    pipe.generate(
        seed=seed,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_path="/path/to/assets/inputs/imgs/img_0.jpg",
        save_result_path=save_result_path,
    )
