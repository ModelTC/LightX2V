"""
Wan2.1 text-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.1 model for T2V generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Wan2.1 T2V task
pipe = LightX2VPipeline(
    model_path="/path/to/Wan2.1-T2V-1.3B",
    model_cls="wan2.1_distill",
    task="t2v",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(config_json="../configs/wan/wan_t2v.json")

pipe.enable_quantize(dit_quantized=True, dit_quantized_ckpt="lightx2v/Wan-NVFP4/wan2.1_t2v_1_3b_nvfp4_lightx2v_4step.safetensors", quant_scheme="nvfp4")

# Create generator with specified parameters
pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=4,
    height=480,  # Can be set to 720 for higher resolution
    width=832,  # Can be set to 1280 for higher resolution
    num_frames=81,
    guidance_scale=1.0,
    sample_shift=5.0,
)

seed = 42
prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
save_result_path = "/path/to/save_results/output.mp4"

pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
