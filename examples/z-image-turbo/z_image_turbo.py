"""Z-Image-Turbo text-to-image generation with FP8 weights and DiT CPU offload."""

from pathlib import Path

from lightx2v import LightX2VPipeline

# Download Tongyi-MAI/Z-Image-Turbo to a local directory.
pipe = LightX2VPipeline(
    model_path="/path/to/Z-Image-Turbo",
    model_cls="z_image",
    task="t2i",
)

# Download FP8 DiT weights from lightx2v/Z-Image-Turbo-Quantized; INT4 Qwen3 is optional.
pipe.enable_quantize(
    dit_quantized=True,
    dit_quantized_ckpt="/path/to/Z-Image-Turbo-Quantized/z_image_turbo_scaled_fp8_e4m3fn.safetensors",
    quant_scheme="fp8-sgl",
    # text_encoder_quantized=True,
    # text_encoder_quantized_ckpt="JunHowie/Qwen3-4B-GPTQ-Int4",
    # text_encoder_quant_scheme="int4"
)

# Offload DiT weights to CPU; the text encoder and VAE remain on GPU.
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="model",  # ["model", "block"]
)

# Choose one: load JSON settings or pass inference parameters below.
config_path = Path(__file__).resolve().parents[2] / "configs/z_image/z_image_turbo_t2i.json"
# pipe.create_generator(config_json=str(config_path))
pipe.create_generator(
    attn_mode="flash_attn3",  # Hopper; use "flash_attn2" where supported.
    size=(480, 832),  # (height, width); use size=() for aspect_ratio presets.
    aspect_ratio="16:9",
    infer_steps=9,
    guidance_scale=1,
)

# Generation parameters
seed = 42
prompt = 'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition, Ultra HD, 4K, cinematic composition.'
save_result_path = "/path/to/save_results/output.png"

# Generate image
pipe.generate(
    seed=seed,
    prompt=prompt,
    save_result_path=save_result_path,
)
