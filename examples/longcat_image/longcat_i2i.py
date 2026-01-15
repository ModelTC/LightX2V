"""
LongCat Image Edit (I2I) generation example.
This example demonstrates how to use LightX2V with LongCat-Image-Edit model for I2I generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for LongCat-Image-Edit I2I task
pipe = LightX2VPipeline(
    model_path="/data/nvme1/models/meituan-longcat/LongCat-Image-Edit",
    model_cls="longcat_image",
    task="i2i",
)

# Create generator from config JSON file
pipe.create_generator(
    config_json="/workspace/configs/longcat_image/longcat_image_i2i.json"
)

# Generation parameters
seed = 43
prompt = "将猫变成狗"
negative_prompt = ""
image_path = "/data/nvme1/models/meituan-longcat/LongCat-Image-Edit/assets/test.png"
save_result_path = "/workspace/save_results/longcat_image_i2i_pipeline.png"

# Generate edited image
pipe.generate(
    seed=seed,
    image_path=image_path,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
