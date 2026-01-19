"""
Wan2.2 distilled model text-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.2 distilled model for T2V generation.
"""

from lightx2v import LightX2VPipeline

pipe = LightX2VPipeline(model_path="/data/nvme0/gushiqiao/models/official_models/wan2.2/Wan2.2-T2V-A14B", model_cls="wan2.2_moe_distill", task="t2v")


pipe.enable_quantize(
    dit_quantized=True,
    high_noise_quantized_ckpt="/data/nvme0/gushiqiao/models/official_models/wan2.2/Wan2.2-T2V-A14B-disilled/wan2.2_t2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_1217.safetensors",
    low_noise_quantized_ckpt="/data/nvme0/gushiqiao/models/official_models/wan2.2/Wan2.2-T2V-A14B-disilled/wan2.2_t2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step_1217.safetensors",
)

# Enable offloading to significantly reduce VRAM usage
# Suitable for RTX 30/40/50 consumer GPUs
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block",
    text_encoder_offload=True,
    image_encoder_offload=False,
    vae_offload=False,
)

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
prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
save_result_path = "./output.mp4"

pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
