"""
Wan2.1 text-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.2-5B model for T2V generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Wan2.2-5b TI2V task
pipe = LightX2VPipeline(
    model_path="/path/to/Wan2.2-TI2V-5B",
    model_cls="wan2.2",
    task="t2v",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(config_json="../configs/wan22/wan_ti2v_t2v.json")

# Create generator with specified parameters
pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=50,
    height=704,
    width=1280,
    num_frames=121,
    fps=25,
    guidance_scale=5.0,
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
