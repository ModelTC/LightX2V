"""
Wan2.1 first-last-frame-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.1 model for FLF2V generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for FLF2V task
pipe = LightX2VPipeline(
    model_path="/path/to/Wan2.1-FLF2V-14B-720P",
    image_path="../assets/inputs/imgs/flf2v_input_first_frame-fs8.png",
    last_frame_path="../assets/inputs/imgs/flf2v_input_last_frame-fs8.png",
    model_cls="wan2.1",
    task="flf2v",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="../configs/wan/wan_flf2v.json"
# )

# Optional: enable offloading to significantly reduce VRAM usage
# Suitable for RTX 30/40/50 consumer GPUs
# pipe.enable_offload(
#     cpu_offload=True,
#     offload_granularity="block",
#     text_encoder_offload=True,
#     image_encoder_offload=False,
#     vae_offload=False,
# )

# Create generator with specified parameters
pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=40,
    height=480,  # Can be set to 720 for higher resolution
    width=832,  # Can be set to 1280 for higher resolution
    num_frames=81,
    guidance_scale=5,
    sample_shift=5.0,
)

seed = 42
prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird’s feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
save_result_path = "/path/to/save_results/output.mp4"

pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
