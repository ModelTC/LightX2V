"""
SeedVR video super-resolution example.

This example demonstrates how to use LightX2V with SeedVR model
for video/image super-resolution (SR) task.
"""

from datetime import datetime
from pathlib import Path

from lightx2v.pipeline import LightX2VPipeline

repo_root = Path(__file__).resolve().parents[3]

ts = datetime.now().strftime("%y%m%d%H%M")

# Initialize pipeline for SeedVR SR task
pipe = LightX2VPipeline(
    model_path="/path/to/ByteDance-Seed/SeedVR2-3B",
    model_cls="seedvr2",
    task="sr",
)

pipe.create_generator(config_json=str(repo_root / "configs/seedvr/seedvr2_3b.json"))

seed = 42
save_result_path = f"output_sr_{ts}.mp4"

# Input video or image path (required for SR task)
input_video_path = "input.mp4"
# Or use an image for single-frame SR
# input_image_path = "/path/to/input_low_res.png"

# Generate super-resolved video
pipe.generate(
    seed=seed,
    save_result_path=save_result_path,
    video_path=input_video_path,
    # Or use image_path for single-frame SR:
    # image_path=input_image_path,
)
