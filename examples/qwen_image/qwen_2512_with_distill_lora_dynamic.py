"""
Qwen-image-edit Text-to-image generation example.
This example demonstrates how to use LightX2V with Qwen-Image-2512 model for T2I generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Qwen-image-2512 T2I task
pipe = LightX2VPipeline(
    model_path="/path/to/Qwen/Qwen-Image-2512",
    model_cls="qwen-image-2512",
    task="t2i",
)


# Enable offloading to significantly reduce VRAM usage with minimal speed impact
# Suitable for RTX 30/40/50 consumer GPUs
# pipe.enable_offload(
#     cpu_offload=True,
#     offload_granularity="block", #["block", "phase"]
#     text_encoder_offload=True,
#     vae_offload=False,
# )

# Load fp8 Base weights (and int4 Qwen2_5 vl model (optional))
pipe.enable_quantize(
    dit_quantized=True,
    dit_quantized_ckpt="lightx2v/Qwen-Image-2512-Lightning/qwen_image_2512_fp8_e4m3fn_scaled.safetensors",
    quant_scheme="fp8-sgl",
    # text_encoder_quantized=True,
    # text_encoder_quantized_ckpt="lightx2v/Encoders/GPTQModel/Qwen25-VL-4bit-GPTQ",
    # text_encoder_quant_scheme="int4"
)

# Load distilled LoRA weights
pipe.enable_lora(
    [
        {
            "path": "lightx2v/Qwen-Image-2512-Lightning/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
            "strength": 1.0,
        },
    ],
    lora_dynamic_apply=True,  # Support inference with LoRA weights, save memory but slower, default is False
)
# Create generator manually with specified parameters
pipe.create_generator(
    attn_mode="flash_attn3",
    aspect_ratio="16:9",
    infer_steps=4,
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
