"""
HunyuanVideo-1.5 image-to-video generation example with quantization.
This example demonstrates how to use LightX2V with HunyuanVideo-1.5 model for I2V generation,
including quantized model usage for reduced memory consumption.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for HunyuanVideo-1.5 I2V task
pipe = LightX2VPipeline(
    image_path="/path/to/assets/inputs/imgs/img_0.jpg",
    model_path="/path/to/ckpts/hunyuanvideo-1.5/",
    model_cls="hunyuan_video_1.5",
    transformer_model_name="720p_i2v",
    task="i2v",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(config_json="../configs/hunyuan_video_15/hunyuan_video_i2v_720p.json")

# Enable offloading to significantly reduce VRAM usage with minimal speed impact
# Suitable for RTX 30/40/50 consumer GPUs
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block",  # For HunyuanVideo-1.5, only "block" is supported
    text_encoder_offload=True,
    image_encoder_offload=False,
    vae_offload=False,
)

# Enable quantization for reduced memory usage
# Quantized models can be downloaded from: https://huggingface.co/lightx2v/Hy1.5-Quantized-Models
pipe.enable_quantize(
    quant_scheme="fp8-sgl",
    dit_quantized=True,
    dit_quantized_ckpt="/path/to/hy15_720p_i2v_fp8_e4m3_lightx2v.safetensors",
    text_encoder_quantized=True,
    image_encoder_quantized=False,
    text_encoder_quantized_ckpt="/path/to/hy15_qwen25vl_llm_encoder_fp8_e4m3_lightx2v.safetensors",
)

# Create generator with specified parameters
pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=50,
    num_frames=121,
    guidance_scale=6.0,
    sample_shift=7.0,
    fps=24,
)

# Generation parameters
seed = 42
prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
negative_prompt = ""
save_result_path = "/path/to/save_results/output2.mp4"

# Generate video
pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
