from lightx2v import LightX2VPipeline

pipe = LightX2VPipeline(model_path="Lightricks/LTX-2/", model_cls="ltx2", task="i2av", dit_original_ckpt="Lightricks/LTX-2/ltx-2-19b-dev.safetensors")

# pipe.enable_offload(
#     cpu_offload=True,
#     offload_granularity="block",
#     text_encoder_offload=False,
#     vae_offload=False,
# )

pipe.create_generator(
    attn_mode="sage_attn2",
    infer_steps=40,
    height=512,
    width=768,
    num_frames=121,
    guidance_scale=4.0,
    sample_shift=[2.05, 0.95],
    fps=24,
    audio_fps=24000,
    double_precision_rope=True,
    norm_modulate_backend="triton",  # "torch"
)

seed = 42
image_path = "/path/to/woman.jpeg"  # For multiple images, use comma-separated paths: "path1.jpg,path2.jpg"
image_strength = 1.0  # Scalar: use same strength for all images, or list: [1.0, 0.8] for different strengths
prompt = "A young woman with wavy, shoulder-length light brown hair is singing and dancing joyfully outdoors on a foggy day. She wears a cozy pink turtleneck sweater, swaying gracefully to the music with animated expressions and bright, piercing blue eyes. Her movements are fluid and energetic as she twirls and gestures expressively. A wooden fence and a misty, grassy field fade into the background, creating a dreamy atmosphere for her lively performance."
negative_prompt = "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
save_result_path = "/path/to/save_results/output.mp4"

# Note: image_strength can also be set in config_json
# For scalar: image_strength = 1.0 (all images use same strength)
# For list: image_strength = [1.0, 0.8] (must match number of images)

pipe.generate(
    seed=seed,
    prompt=prompt,
    image_path=image_path,
    image_strength=image_strength,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
