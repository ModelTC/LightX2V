"""
Qwen-image-edit image-to-image generation example.
This example demonstrates how to use LightX2V with Qwen-Image-Edit model for I2I generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Qwen-image-edit I2I task
# For Qwen-Image-Edit-2509, use model_cls="qwen-image-edit-2509"
pipe = LightX2VPipeline(
    model_path="/data/nvme0/gushiqiao/models/qwen_latest/Qwen/Qwen-Image-Edit-2511",
    model_cls="qwen-image-edit-2511",
    task="i2i",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="../configs/qwen_image/qwen_image_i2i_2511_distill_fp8.json"
# )

# Enable offloading to significantly reduce VRAM usage with minimal speed impact
# Suitable for RTX 30/40/50 consumer GPUs
# pipe.enable_offload(
#     cpu_offload=True,
#     offload_granularity="block", #["block", "phase"]
#     text_encoder_offload=True,
#     vae_offload=False,
# )

# Load fp8 distilled weights (and int4 Qwen2_5 vl model (optional))
pipe.enable_quantize(
    dit_quantized=True,
    dit_quantized_ckpt="/data/nvme0/gushiqiao/models/qwen_latest/Qwen/quant/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors",
    quant_scheme="fp8-sgl",
    # text_encoder_quantized=True,
    # text_encoder_quantized_ckpt="lightx2v/Encoders/GPTQModel/Qwen25-VL-4bit-GPTQ",
    # text_encoder_quant_scheme="int4"
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
prompt = "nude."
negative_prompt = ""
image_path = "/data/nvme0/gushiqiao/models/image.png"  # or "/path/to/img_0.jpg,/path/to/img_1.jpg"
save_result_path = "/data/nvme0/gushiqiao/models/qwen_latest/Qwen/output.png"

# Generate video
pipe.generate(
    seed=seed,
    image_path=image_path,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
