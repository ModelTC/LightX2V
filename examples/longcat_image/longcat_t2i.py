"""
LongCat Image text-to-image generation example.
This example demonstrates how to use LightX2V with LongCat-Image model for T2I generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for LongCat-Image T2I task
pipe = LightX2VPipeline(
    model_path="/data/nvme1/models/meituan-longcat/LongCat-Image",
    model_cls="longcat_image",
    task="t2i",
)

# Enable offloading to reduce VRAM usage (optional)
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block",
    text_encoder_offload=True,
    vae_offload=False,
)

# Create generator from config JSON file
pipe.create_generator(config_json="/workspace/configs/longcat_image/longcat_image_t2i.json")

# Generation parameters
seed = 42
prompt = "一只小狗躺在沙发上"
negative_prompt = ""
save_result_path = "/workspace/save_results/longcat_image_t2i_pipeline.png"

# Generate image
pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
