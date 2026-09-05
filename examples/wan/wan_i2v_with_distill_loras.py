"""
Wan2.2 distilled model with LoRA image-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.2 distilled model and LoRA for I2V generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Wan2.2 distilled I2V task with LoRA
# For Wan2.1, use model_cls="wan2.1" with distill_method="dmd2".
pipe = LightX2VPipeline(
    model_path="/path/to/wan2.2/Wan2.2-I2V-A14B",
    model_cls="wan2.2_moe",
    task="i2v",
    distill_method="dmd2",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="configs/distill/wan22/wan_moe_i2v_distill_with_lora.json"
# )

# Enable offloading to significantly reduce VRAM usage
# Suitable for RTX 30/40/50 consumer GPUs
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block",
    text_encoder_offload=True,
    image_encoder_offload=False,
    vae_offload=False,
)

# Load distilled LoRA weights
pipe.enable_lora(
    [
        {"name": "high_noise_model", "path": "lightx2v/Wan2.2-Distill-Loras/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors", "strength": 1.0},
        {"name": "low_noise_model", "path": "lightx2v/Wan2.2-Distill-Loras/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors", "strength": 1.0},
    ],
    lora_dynamic_apply=False,  # Support inference with LoRA weights, save memory but slower, default is False
)

# Create generator with specified parameters
pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=4,
    height=480,  # Can be set to 720 for higher resolution
    width=832,  # Can be set to 1280 for higher resolution
    num_frames=81,
    guidance_scale=1,
    sample_shift=5.0,
)

seed = 42
prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
save_result_path = "/path/to/save_results/output.mp4"
image_path = "/path/to/img_0.jpg"

pipe.generate(
    seed=seed,
    image_path=image_path,
    prompt=prompt,
    save_result_path=save_result_path,
)
