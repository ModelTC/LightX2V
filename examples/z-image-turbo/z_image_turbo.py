"""
Z-Image image-to-image generation example.
This example demonstrates how to use LightX2V with Z-Image-Turbo model for T2I generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Z-Image-edit T2I task
pipe = LightX2VPipeline(
    model_path="Tongyi-MAI/Z-Image-Turbo",
    model_cls="z_image",
    task="t2i",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="../configs/z_image/z_image_turbo_t2i.json"
# )

# Load fp8 distilled weights (and int4 Qwen3 model (optional))
pipe.enable_quantize(
    dit_quantized=True,
    dit_quantized_ckpt="lightx2v/Z-Image-Turbo-Quantized/z_image_turbo_scaled_fp8_e4m3fn.safetensors",
    quant_scheme="fp8-sgl",
    # text_encoder_quantized=True,
    # text_encoder_quantized_ckpt="JunHowie/Qwen3-4B-GPTQ-Int4",
    # text_encoder_quant_scheme="int4"
)

# Enable offloading to significantly reduce VRAM usage with minimal speed impact
# Suitable for RTX 30/40/50 consumer GPUs
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="model",  # ["model", "block"]
)

# Create generator manually with specified parameters
pipe.create_generator(
    attn_mode="flash_attn3",
    aspect_ratio="16:9",
    infer_steps=9,
    guidance_scale=1,
)

# Generation parameters
seed = 42
prompt = 'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition, Ultra HD, 4K, cinematic composition.'
negative_prompt = ""
save_result_path = "/path/to/save_results/output.png"

# Generate video
pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
