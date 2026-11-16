"""
HunyuanVideo-1.5 text-to-video generation example.
This example demonstrates how to use LightX2V with HunyuanVideo-1.5 4-step distilled model for T2V generation.
"""

import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads((Path(__file__).resolve().parents[3] / "configs/hunyuan_video_15/hunyuan_video_t2v_480p_distill.json").read_text())
config.update(
    {
        "model_path": "/path/to/ckpts/hunyuanvideo-1.5/",
        "model_cls": "hunyuan_video_1.5",
        "task": "t2v",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 123
    prompt = "A close-up shot captures a scene on a polished, light-colored granite kitchen counter, illuminated by soft natural light from an unseen window. Initially, the frame focuses on a tall, clear glass filled with golden, translucent apple juice standing next to a single, shiny red apple with a green leaf still attached to its stem. The camera moves horizontally to the right. As the shot progresses, a white ceramic plate smoothly enters the frame, revealing a fresh arrangement of about seven or eight more apples, a mix of vibrant reds and greens, piled neatly upon it. A shallow depth of field keeps the focus sharply on the fruit and glass, while the kitchen backsplash in the background remains softly blurred. The scene is in a realistic style."
    save_result_path = "./output_hunyuan_video_15_t2v_distill.mp4"

    pipe.generate(
        seed=seed,
        prompt=prompt,
        save_result_path=save_result_path,
    )
