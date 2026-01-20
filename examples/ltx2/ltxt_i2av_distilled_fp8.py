from lightx2v import LightX2VPipeline

pipe = LightX2VPipeline(
    model_path="Lightricks/LTX-2/",
    model_cls="ltx2",
    task="i2av",
)

pipe.enable_quantize(
    dit_quantized=True,
    dit_quantized_ckpt="Lightricks/LTX-2/ltx-2-19b-distilled-fp8.safetensors",
    quant_scheme="fp8-pertensor",
    skip_fp8_block_index=[0, 43, 44, 45, 46, 47],
)

# pipe.enable_offload(
#     cpu_offload=True,
#     offload_granularity="block",
#     text_encoder_offload=False,
#     vae_offload=False,
# )

pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=8,
    height=512,
    width=768,
    num_frames=121,
    guidance_scale=1.0,
    sample_shift=[2.05, 0.95],
    fps=24,
    audio_fps=24000,
    double_precision_rope=True,
    norm_modulate_backend="triton",  # "torch"
)

seed = 42
images = "/path/to/woman.jpeg:0:1.0"
prompt = "A young woman with wavy, shoulder-length light brown hair stands outdoors on a foggy day. She wears a cozy pink turtleneck sweater, with a serene expression and piercing blue eyes. A wooden fence and a misty, grassy field fade into the background, evoking a calm and introspective mood."
negative_prompt = "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
save_result_path = "/path/to/save_results/output.mp4"

pipe.generate(
    seed=seed,
    prompt=prompt,
    images=images,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
