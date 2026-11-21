import json
from pathlib import Path

from lightx2v import LightX2VPipeline

config = json.loads((Path(__file__).resolve().parents[3] / "configs/ltx2/ltx2.json").read_text())
config.update(
    {
        "model_path": "Lightricks/LTX-2/",
        "model_cls": "ltx2",
        "task": "i2av",
    }
)

with LightX2VPipeline(config) as pipe:
    seed = 42
    image_path = ["/path/to/woman.jpeg"]  # Add further paths as list items.
    image_strength = [1.0]  # One value broadcasts to every image; otherwise provide one per image.
    # Omit frame indices to space the images evenly across the configured clip.
    # Example for 3 images and num_frames=121: omit image_frame_indices to get ~[0, 60, 120], or set explicitly:
    # image_frame_indices = [0, 40, 120]
    image_frame_indices = None
    prompt = "A young woman with wavy, shoulder-length light brown hair is singing and dancing joyfully outdoors on a foggy day. She wears a cozy pink turtleneck sweater, swaying gracefully to the music with animated expressions and bright, piercing blue eyes. Her movements are fluid and energetic as she twirls and gestures expressively. A wooden fence and a misty, grassy field fade into the background, creating a dreamy atmosphere for her lively performance."
    negative_prompt = "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
    save_result_path = "/path/to/save_results/output.mp4"

    # Note: image_strength can also be set in config_json
    # A one-item list, e.g. image_strength = [1.0], applies to every image.
    # For list: image_strength = [1.0, 0.8] (must match number of images)
    # image_frame_indices: list of pixel-frame indices (one per image), or None for even spacing across the clip

    pipe.generate(
        seed=seed,
        prompt=prompt,
        image_path=image_path,
        image_strength=image_strength,
        image_frame_indices=image_frame_indices,
        negative_prompt=negative_prompt,
        save_result_path=save_result_path,
    )
