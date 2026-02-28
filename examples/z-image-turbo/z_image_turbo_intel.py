"""
Z-Image image-to-image generation example.
This example demonstrates how to use LightX2V with Z-Image-Turbo model for T2I generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Z-Image-edit T2I task
pipe = LightX2VPipeline(
    model_path=r"D:\valencia\models\Z-Image-Turbo",
    model_cls="z_image",
    task="t2i",
)

# Alternative: create generator from config JSON file
pipe.create_generator(
    config_json="../../configs/platforms/intel_xpu/z_image_turbo_t2i.json"
)


# Create generator manually with specified parameters
pipe.create_generator(
    attn_mode="torch_sdpa",

    aspect_ratio="16:9",
    infer_steps=9,
    guidance_scale=1,
)

# Generation parameters
seed = 42
prompt = 'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition, Ultra HD, 4K, cinematic composition.'
negative_prompt = ""
save_result_path = "./output.png"
print("generate!!!!!")
import time
s= time.time()
# Generate video
pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
e=time.time()
print("e2e_perf",e-s)

print("success!!!")
