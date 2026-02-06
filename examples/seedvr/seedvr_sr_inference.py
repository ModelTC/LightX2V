"""
SeedVR video super-resolution example.

This example demonstrates how to use LightX2V with SeedVR model
for video/image super-resolution (SR) task.
"""

from datetime import datetime

ts = datetime.now().strftime("%y%m%d%H%M")
from lightx2v.pipeline import LightX2VPipeline

# Initialize pipeline for SeedVR SR task
pipe = LightX2VPipeline(
    model_path="/data/nvme1/models/ByteDance-Seed/SeedVR2-3B/seedvr2_ema_3b.pth",
    # vae_path="/data/nvme1/models/ByteDance-Seed/SeedVR2-3B/ema_vae.pth",
    # pos_emb_path="/data/nvme1/models/ByteDance-Seed/SeedVR2-3B/pos_emb.pt",
    # neg_emb_path="/data/nvme1/models/ByteDance-Seed/SeedVR2-3B/neg_emb.pt",
    model_cls="seedvr2.3b",
    task="sr",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(config_json="../configs/seedvr/seedvr2_3b.json")

# Create generator with specified parameters

pipe.create_generator(config_json="/data/nvme0/chendingyu/LightX2V/configs/seedvr/seedvr2_3b.json")

seed = 42
prompt = "A cinematic video of a sunset over the ocean with golden reflections"
negative_prompt = ""
save_result_path = f"/data/nvme0/chendingyu/SeedVR/output/output_sr_{ts}.mp4"

# Input video or image path (required for SR task)
input_video_path = "/data/nvme0/chendingyu/SeedVR/input/wan2.1_i2v_gpu1_260115074239_svg_sla.mp4"
input_video_path = "/data/nvme0/chendingyu/SeedVR/input/wan2.2_moe_t2v_2601261239.mp4"
# Or use an image for single-frame SR
# input_image_path = "/path/to/input_low_res.png"

# Generate super-resolved video
pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
    video_path=input_video_path,  # Use video_path for video SR
    # Or use image_path for single-frame SR:
    # image_path=input_image_path,
)
