"""
HunyuanVideo-1.5 text-to-video generation example.
This example demonstrates how to use LightX2V with HunyuanVideo-1.5 model for T2V generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for HunyuanVideo-1.5
pipe = LightX2VPipeline(
    model_path="/path/to/ckpts/hunyuanvideo-1.5/",
    model_cls="hunyuan_video_1.5",
    transformer_model_name="720p_t2v",
    task="t2v",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(config_json="../configs/hunyuan_video_15/hunyuan_video_t2v_720p.json")

# Enable offloading to significantly reduce VRAM usage with minimal speed impact
# Suitable for RTX 30/40/50 consumer GPUs
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block",  # For HunyuanVideo-1.5, only "block" is supported
    text_encoder_offload=True,
    image_encoder_offload=False,
    vae_offload=False,
)

# Use lighttae
pipe.enable_lightvae(
    use_tae=True,
    tae_path="/path/to/lighttaehy1_5.safetensors",
    use_lightvae=False,
    vae_path=None,
)

# Create generator with specified parameters
pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=50,
    num_frames=121,
    guidance_scale=6.0,
    sample_shift=9.0,
    aspect_ratio="16:9",
    fps=24,
)

# Generation parameters
seed = 123
prompt = "A close-up shot captures a scene on a polished, light-colored granite kitchen counter, illuminated by soft natural light from an unseen window. Initially, the frame focuses on a tall, clear glass filled with golden, translucent apple juice standing next to a single, shiny red apple with a green leaf still attached to its stem. The camera moves horizontally to the right. As the shot progresses, a white ceramic plate smoothly enters the frame, revealing a fresh arrangement of about seven or eight more apples, a mix of vibrant reds and greens, piled neatly upon it. A shallow depth of field keeps the focus sharply on the fruit and glass, while the kitchen backsplash in the background remains softly blurred. The scene is in a realistic style."
negative_prompt = ""
save_result_path = "/path/to/save_results/output.mp4"

# Generate video
pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
