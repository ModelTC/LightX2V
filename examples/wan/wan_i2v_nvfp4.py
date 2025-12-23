"""
Wan2.1 image-to-video generation example.
This example demonstrates how to use LightX2V with Wan2. model for I2V generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Wan2.2 I2V task
# For wan2.1, use model_cls="wan2.1"
pipe = LightX2VPipeline(
    model_path="/path/to/Wan2.1-I2V-14B-480P",
    model_cls="wan2.1_distill",
    task="i2v",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="../configs/wan22/wan_moe_i2v.json"
# )

# Enable offloading to significantly reduce VRAM usage with minimal speed impact
# Suitable for RTX 30/40/50 consumer GPUs
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block",  # For Wan models, supports both "block" and "phase"
    text_encoder_offload=True,
    image_encoder_offload=False,
    vae_offload=False,
)

pipe.enable_quantize(dit_quantized=True, dit_quantized_ckpt="lightx2v/Wan-NVFP4/wan2.1_i2v_480p_nvfp4_lightx2v_4step.safetensors", quant_scheme="nvfp4")

# Create generator manually with specified parameters
pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=4,
    height=480,  # Can be set to 720 for higher resolution
    width=832,  # Can be set to 1280 for higher resolution
    num_frames=81,
    guidance_scale=1.0,  # For wan2.1, guidance_scale is a scalar (e.g., 5.0)
    sample_shift=5.0,
)

# Generation parameters
seed = 42
prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
image_path = "/path/to/img_0.jpg"
save_result_path = "/path/to/save_results/output.mp4"

# Generate video
pipe.generate(
    seed=seed,
    image_path=image_path,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
