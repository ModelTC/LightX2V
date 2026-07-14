"""
Qwen-image-edit image-to-image generation example.
This example demonstrates how to use LightX2V with Qwen-Image-Edit model for I2I generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Qwen-image-edit I2I task
# For Qwen-Image-Edit-2509, use model_cls="qwen-image-edit-2509"
pipe = LightX2VPipeline(
    model_path="/path/to/Qwen-Image-Edit-2511",
    model_cls="qwen-image-edit-2511",
    task="i2i",
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
    dit_quantized_ckpt="lightx2v/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors",
    quant_scheme="fp8-sgl",
    # text_encoder_quantized=True,
    # text_encoder_quantized_ckpt="lightx2v/Encoders/GPTQModel/Qwen25-VL-4bit-GPTQ",
    # text_encoder_quant_scheme="int4"
)

# Load distilled LoRA weights
pipe.enable_lora(
    [
        {
            "path": "lightx2v/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
            "strength": 1.0,
        },
    ],
    lora_dynamic_apply=True,  # Support inference with LoRA weights, save memory but slower, default is False
)
# Create generator manually with specified parameters
pipe.create_generator(
    attn_mode="flash_attn3",
    resize_mode="adaptive",
    infer_steps=8,
    guidance_scale=1,
)

# Generation parameters
seed = 42
prompt = "Replace the polka-dot shirt with a light blue shirt."
negative_prompt = ""
image_path = "/path/to/img.png"  # or "/path/to/img_0.jpg,/path/to/img_1.jpg"
save_result_path = "/path/to/save_results/output.png"

# Generate video
pipe.generate(
    seed=seed,
    image_path=image_path,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
